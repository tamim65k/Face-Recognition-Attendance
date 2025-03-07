import cv2
import dlib
import os
import numpy as np
import pickle
import csv
from datetime import datetime
from scipy.spatial import distance as dist
from skimage import feature
from collections import defaultdict

# Load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load the trained SVM model and LabelEncoder
with open("face_recognition_svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Indexes for eye landmarks
LEFT_EYE_INDEXES = list(range(36, 42))
RIGHT_EYE_INDEXES = list(range(42, 48))

# Dictionary to store the status of recognized faces
recognized_faces = defaultdict(lambda: "Unknown")
blinks_detected = defaultdict(int)  # Track the number of blinks detected
eye_positions = defaultdict(lambda: None)  # Track previous eye positions
current_face_keys = set()  # Track current face keys
attendance_records = set()  # Track attendance to avoid duplicates

# Initialize the attendance directory
attendance_dir = "attendance"
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def extract_embeddings(img, face_rect):
    shape = predictor(img, face_rect)
    face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
    return np.array(face_descriptor)

def detect_blinks(shape):
    left_eye = shape[LEFT_EYE_INDEXES]
    right_eye = shape[RIGHT_EYE_INDEXES]
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0
    return ear

def detect_eye_movement(shape, prev_eye_pos):
    left_eye = shape[LEFT_EYE_INDEXES]
    right_eye = shape[RIGHT_EYE_INDEXES]
    eye_center = np.mean([np.mean(left_eye, axis=0), np.mean(right_eye, axis=0)], axis=0)
    if prev_eye_pos is None:
        return eye_center, True
    movement = np.linalg.norm(eye_center - prev_eye_pos)
    return eye_center, movement > 1  # Adjust threshold accordingly

def detect_head_movement(shape, prev_shape):
    if prev_shape is None:
        return shape, False
    movement = np.linalg.norm(shape - prev_shape)
    return shape, movement > 10  # Adjust threshold accordingly

def analyze_texture(face_image):
    if face_image is None or face_image.size == 0:
        return None  # Return None if face_image is empty
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, 24, 8, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def recognize_faces(frame, prev_shapes, eye_positions):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    current_shapes = []
    current_face_keys.clear()  # Clear the current face keys for this frame

    for face_rect in faces:
        shape = predictor(gray, face_rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        current_shapes.append(shape)

        ear = detect_blinks(shape)
        prev_shape, head_movement_detected = None, False
        for ps in prev_shapes:
            _, movement_detected = detect_head_movement(shape, ps)
            if movement_detected:
                prev_shape = ps
                head_movement_detected = True
                break

        eye_center, eye_movement_detected = detect_eye_movement(shape, eye_positions.get(tuple(shape.flatten())))
        eye_positions[tuple(shape.flatten())] = eye_center

        x, y, w, h = (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height())
        embedding = extract_embeddings(frame, face_rect)
        embedding = np.expand_dims(embedding, axis=0)

        yhat_class = svm_model.predict(embedding)
        yhat_prob = svm_model.predict_proba(embedding)

        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100

        predict_name = encoder.inverse_transform(yhat_class)[0]
        label = f"{predict_name} ({class_probability:.2f}%)"

        face_image = frame[y:y + h, x:x + w]

        if face_image is None or face_image.size == 0:
            print(f"Skipped empty face image at ({x}, {y}, {w}, {h})")
            continue  # Skip this face if the image is empty

        texture_hist = analyze_texture(face_image)

        # Only check texture_score if texture_hist is not None
        if texture_hist is not None:
            # Checking if face texture matches typical face texture patterns
            texture_score = np.sum((texture_hist - 0.1) ** 2)
            texture_threshold = 0.5  # Adjust texture threshold accordingly

            threshold = 60  # Minimum confidence threshold for recognition
            face_key = tuple(embedding.flatten())  # Using embedding as a unique key for the face

            current_face_keys.add(face_key)  # Add face key to current set

            if recognized_faces[face_key] in ["Unknown", "Fake"]:
                if ear < 0.25:
                    blinks_detected[face_key] += 1

                if blinks_detected[face_key] >= 1 and head_movement_detected and eye_movement_detected and texture_score < texture_threshold:
                    if class_probability > threshold and predict_name != "unknown":
                        recognized_faces[face_key] = "Real"
                        # Check if attendance has already been recorded
                        if predict_name not in attendance_records:
                            # Record attendance with current date and time
                            now = datetime.now()
                            date_str = now.strftime("%Y-%m-%d")
                            time_str = now.strftime("%H:%M:%S")
                            csv_file = os.path.join(attendance_dir, f"{date_str}.csv")
                            if not os.path.exists(csv_file):
                                with open(csv_file, 'x', newline='') as file:
                                    writer = csv.writer(file)
                                    writer.writerow(["Name", "Date", "Time"])
                            with open(csv_file, 'a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([predict_name, date_str, time_str])
                            attendance_records.add(predict_name)
                    else:
                        recognized_faces[face_key] = "Fake"

            if class_probability < threshold:
                predict_name = "unknown"
                label = "unknown"

            status = recognized_faces[face_key]
            color = (0, 255, 255) if status == "Real" else (0, 0, 255)

            # Display the text in the desired format
            cv2.putText(frame, status, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"{predict_name} ({class_probability:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame, current_shapes, eye_positions

def start_recognition():
    cam = cv2.VideoCapture(2)
    prev_shapes = []
    eye_positions = {}
    prev_face_keys = set()  # Track previous frame face keys

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame, prev_shapes, eye_positions = recognize_faces(frame, prev_shapes, eye_positions)

        # If new faces are detected in the current frame, reset the status
        if not prev_face_keys.issubset(current_face_keys):
            recognized_faces.clear()
            blinks_detected.clear()
            eye_positions.clear()

        prev_face_keys = current_face_keys.copy()  # Update the previous face keys
        cv2.imshow('Real-Time Face Recognition with Liveness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_recognition()
