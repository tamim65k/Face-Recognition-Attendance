import os
import cv2
import numpy as np
import pickle
import dlib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Load the face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def ensure_directories():
    known_faces_dir = "KnownFaces"
    unknown_faces_dir = "UnknownFaces"
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
        print(f"Created directory: {known_faces_dir}")
    if not os.path.exists(unknown_faces_dir):
        os.makedirs(unknown_faces_dir)
        print(f"Created directory: {unknown_faces_dir}")

def extract_embeddings(img, face_rect):
    """
    Extracts 128-dimensional face embeddings using dlib.
    """
    shape = predictor(img, face_rect)
    face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
    return np.array(face_descriptor)

def train_model():
    ensure_directories()
    db_path = "KnownFaces"
    unknown_db_path = "UnknownFaces"
    embeddings = []

    # Process known faces
    for person_name in os.listdir(db_path):
        person_folder = os.path.join(db_path, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    faces = detector(image, 1)
                    if faces:
                        try:
                            face_rect = faces[0]  # Use the first detected face
                            embedding = extract_embeddings(image, face_rect)
                            embeddings.append((embedding, person_name))
                            print(f"Processed {image_path} for {person_name}")
                        except Exception as e:
                            print(f"Error processing {image_path}: {e}")
                else:
                    print(f"Could not read image: {image_path}")

    # Process unknown faces
    for image_name in os.listdir(unknown_db_path):
        image_path = os.path.join(unknown_db_path, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            faces = detector(image, 1)
            if faces:
                try:
                    face_rect = faces[0]  # Use the first detected face
                    embedding = extract_embeddings(image, face_rect)
                    embeddings.append((embedding, "unknown"))
                    print(f"Processed {image_path} for unknown")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
        else:
            print(f"Could not read image: {image_path}")

    X, y = zip(*embeddings)
    X = np.array(X)
    y = np.array(y)

    # Encode the labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Train the SVM model
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X, y)

    # Save the model and encoder separately
    with open("face_recognition_svm_model.pkl", "wb") as f:
        pickle.dump(svm_model, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print("SVM classifier training complete.")
    print(f"Generated embeddings for {len(embeddings)} images from {len(set([person_name for _, person_name in embeddings]))} classes.")

if __name__ == "__main__":
    train_model()
