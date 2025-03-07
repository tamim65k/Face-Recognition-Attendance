# main gui
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tkinter import simpledialog
import customtkinter
import cv2,os,csv
import numpy as np
import time
from scipy.spatial import distance as dist
import dlib
import pickle
from datetime import datetime
from skimage import feature
from collections import defaultdict


customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light")
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Set the custom icon 
        self.iconbitmap("face-detection.ico")

        # Load the last used ipcam value from a file
        self.ipcam = 2
        
        # Load the face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

        # Load the trained SVM model and LabelEncoder
        with open("face_recognition_svm_model.pkl", "rb") as f:
            self.svm_model = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            self.encoder = pickle.load(f)
        
        

        # Indexes for eye landmarks
        self.LEFT_EYE_INDEXES = list(range(36, 42))
        self.RIGHT_EYE_INDEXES = list(range(42, 48))

        # Dictionary to store the status of recognized faces
        self.recognized_faces = defaultdict(lambda: "Unknown")
        self.blinks_detected = defaultdict(int)  # Track the number of blinks detected
        self.eye_positions = defaultdict(lambda: None)  # Track previous eye positions
        self.current_face_keys = set()  # Track current face keys
        self.attendance_records = set()  # Track attendance to avoid duplicates

        # Initialize the attendance directory
        self.attendance_dir = "attendance"
        if not os.path.exists(self.attendance_dir):
            os.makedirs(self.attendance_dir)

        # configure window
        self.title("Face Recognition-Based")
        self.geometry(f"{800}x{450}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Attendance System", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Add current time label
        self.time_label = customtkinter.CTkLabel(self.sidebar_frame, font=customtkinter.CTkFont(size=16))
        self.time_label.grid(row=1, column=0, padx=20, pady=(0, 10))
        self.update_time()

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame,fg_color="green", text="Take Attendance", command=self.take_attendance)
        self.sidebar_button_1.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame,fg_color="green", text="New Registration", command=self.new_registration)
        self.sidebar_button_2.grid(row=3, column=0, padx=20, pady=10)

        self.dev_settings_button = customtkinter.CTkButton(self.sidebar_frame,width=100, text="Dev Settings", command=self.dev_settings)
        self.dev_settings_button.grid(row=5, column=0, padx=0, pady=(0,30),)
        

        # # Appearance mode widgets
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        # self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                        command=self.change_appearance_mode_event)
        # self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        # self.appearance_mode_optionemenu.set("Light")

        # Open the Mark Attendance frame by default
        self.take_attendance()

    def update_time(self):
        current_time = time.strftime("%d/%m/%Y %H:%M:%S")
        self.time_label.configure(text=current_time)
        self.after(1000, self.update_time)  # Update time every second

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def sidebar_button_event(self):
        print("sidebar_button click")

    def take_attendance(self):
        self.sidebar_button_1.grid_remove()
        self.sidebar_button_2.grid()
        self.custom_frame = customtkinter.CTkFrame(self, height=500)
        self.custom_frame.grid(row=0, column=1, padx=(30, 30), pady=(30, 30), sticky="nsew")
        
        # Add a note in the Mark Attendance frame
        note_text = ("For taking attendance, follow these instructions:\n\n"
                     "1. First, add new user details for taking attendance.\n"
                     "2. Then click 'Capture Snapshot' and select the video of the user. "
                     "The video should be a maximum of 10 seconds.\n"
                     "3. Then click 'Save Details'.\n"
                     "4. Before taking attendance select your camera index from dev settings.\n\n"
                     "Now you can take attendance of the user by clicking 'Mark Attendance'.")
        note_label = customtkinter.CTkLabel(self.custom_frame, text=note_text, font=('times', 18), wraplength=450, justify="left")
        note_label.pack(pady=20)

        self.take_att = customtkinter.CTkButton(self.custom_frame, fg_color="green", text="Mark Attendance", command=self.start_recognition)
        self.take_att.pack(pady=20)


#33333333333333333 mark attedance 333333333333



    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def extract_embeddings(self, img, face_rect):
        shape = self.predictor(img, face_rect)
        face_descriptor = self.face_rec_model.compute_face_descriptor(img, shape)
        return np.array(face_descriptor)

    def detect_blinks(self, shape):
        left_eye = shape[self.LEFT_EYE_INDEXES]
        right_eye = shape[self.RIGHT_EYE_INDEXES]
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        return ear

    def detect_eye_movement(self, shape, prev_eye_pos):
        left_eye = shape[self.LEFT_EYE_INDEXES]
        right_eye = shape[self.RIGHT_EYE_INDEXES]
        eye_center = np.mean([np.mean(left_eye, axis=0), np.mean(right_eye, axis=0)], axis=0)
        if prev_eye_pos is None:
            return eye_center, True
        movement = np.linalg.norm(eye_center - prev_eye_pos)
        return eye_center, movement > 1  # Adjust threshold accordingly

    def detect_head_movement(self, shape, prev_shape):
        if prev_shape is None:
            return shape, False
        movement = np.linalg.norm(shape - prev_shape)
        return shape, movement > 10  # Adjust threshold accordingly

    def analyze_texture(self, face_image):
        if face_image is None or face_image.size == 0:
            return None  # Return None if face_image is empty
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, 24, 8, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
        hist = hist.astype("float")
        hist /= hist.sum()
        return hist

    def recognize_faces(self, frame, prev_shapes, eye_positions):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        current_shapes = []
        self.current_face_keys.clear()  # Clear the current face keys for this frame

        for face_rect in faces:
            shape = self.predictor(gray, face_rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            current_shapes.append(shape)

            ear = self.detect_blinks(shape)
            prev_shape, head_movement_detected = None, False
            for ps in prev_shapes:
                _, movement_detected = self.detect_head_movement(shape, ps)
                if movement_detected:
                    prev_shape = ps
                    head_movement_detected = True
                    break

            eye_center, eye_movement_detected = self.detect_eye_movement(shape, eye_positions.get(tuple(shape.flatten())))
            eye_positions[tuple(shape.flatten())] = eye_center

            x, y, w, h = (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height())
            embedding = self.extract_embeddings(frame, face_rect)
            embedding = np.expand_dims(embedding, axis=0)

            yhat_class = self.svm_model.predict(embedding)
            yhat_prob = self.svm_model.predict_proba(embedding)

            class_index = yhat_class[0]
            class_probability = yhat_prob[0, class_index] * 100

            predict_name = self.encoder.inverse_transform(yhat_class)[0]
            label = f"{predict_name} ({class_probability:.2f}%)"

            face_image = frame[y:y + h, x:x + w]

            if face_image is None or face_image.size == 0:
                print(f"Skipped empty face image at ({x}, {y}, {w}, {h})")
                continue  # Skip this face if the image is empty

            texture_hist = self.analyze_texture(face_image)

            # Only check texture_score if texture_hist is not None
            if texture_hist is not None:
                # Checking if face texture matches typical face texture patterns
                texture_score = np.sum((texture_hist - 0.1) ** 2)
                texture_threshold = 0.5  # Adjust texture threshold accordingly

                threshold = 60  # Minimum confidence threshold for recognition
                face_key = tuple(embedding.flatten())  # Using embedding as a unique key for the face

                self.current_face_keys.add(face_key)  # Add face key to current set

                if self.recognized_faces[face_key] in ["Unknown", "Fake"]:
                    if ear < 0.25:
                        self.blinks_detected[face_key] += 1

                    if self.blinks_detected[face_key] >= 1 and head_movement_detected and eye_movement_detected and texture_score < texture_threshold:
                        if class_probability > threshold and predict_name != "unknown":
                            self.recognized_faces[face_key] = "Real"
                            # Check if attendance has already been recorded
                            if predict_name not in self.attendance_records:
                                # Record attendance with current date and time
                                now = datetime.now()
                                date_str = now.strftime("%Y-%m-%d")
                                time_str = now.strftime("%H:%M:%S")
                                csv_file = os.path.join(self.attendance_dir, f"{date_str}.csv")
                                if not os.path.exists(csv_file):
                                    try:
                                        with open(csv_file, 'w', newline='') as file:
                                            writer = csv.writer(file)
                                            writer.writerow(["Name", "Date", "Time"])
                                    except Exception as e:
                                        print(f"Error creating file {csv_file}: {e}")
                                try:
                                    with open(csv_file, 'a', newline='') as file:
                                        writer = csv.writer(file)
                                        writer.writerow([predict_name, date_str, time_str])
                                    self.attendance_records.add(predict_name)
                                except Exception as e:
                                    print(f"Error writing to file {csv_file}: {e}")
                        else:
                            self.recognized_faces[face_key] = "Fake"

                if class_probability < threshold:
                    predict_name = "unknown"
                    label = "unknown"

                status = self.recognized_faces[face_key]
                color = (0, 255, 255) if status == "Real" else (0, 0, 255)

                # Display the text in the desired format
                cv2.putText(frame, status, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"{predict_name} ({class_probability:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        return frame, current_shapes, eye_positions


    def start_recognition(self):
        cam = cv2.VideoCapture(2)
        prev_shapes = []
        self.eye_positions = {}
        prev_face_keys = set()  # Track previous frame face keys

        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame, prev_shapes, self.eye_positions = self.recognize_faces(frame, prev_shapes, self.eye_positions)

            # If new faces are detected in the current frame, reset the status
            if not prev_face_keys.issubset(self.current_face_keys):
                self.recognized_faces.clear()
                self.blinks_detected.clear()
                self.eye_positions.clear()

            prev_face_keys = self.current_face_keys.copy()  # Update the previous face keys
            cv2.imshow('Real-Time Face Recognition with Liveness Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()




#333333333333333333333333333333333333333333    

    def new_registration(self):
        self.sidebar_button_2.grid_remove()
        self.sidebar_button_1.grid()

        self.custom_frame = customtkinter.CTkFrame(self, height=400)  # Increased width for more space
        self.custom_frame.grid(row=0, column=1, padx=(30, 30), pady=(30, 30), sticky="nsew")

        # # ID Label and Entry
        # id_label = customtkinter.CTkLabel(self.custom_frame, font=('times', 20), text="ID")
        # id_label.grid(row=0, column=0, padx=(50, 0), pady=(45, 0))
        
        # self.id_entry = customtkinter.CTkEntry(self.custom_frame, width=300, placeholder_text="enter your id")  # Increased width
        # self.id_entry.grid(row=0, column=1, padx=(20, 0), pady=(45, 0))

        # Name Label and Entry
        name_label = customtkinter.CTkLabel(self.custom_frame, font=('times', 20), text="NAME")
        name_label.grid(row=1, column=0, padx=(50, 0), pady=(80, 0))
        
        self.name_entry = customtkinter.CTkEntry(self.custom_frame, width=300, placeholder_text="enter your nick name")  # Increased width
        self.name_entry.grid(row=1, column=1, padx=(20, 0), pady=(80, 0))

        # Take Image and Save Profile Buttons
        take_image_button = customtkinter.CTkButton(self.custom_frame,fg_color="green", width=100, text="Capture Snapshot", command=self.capture_snapshot)
        take_image_button.grid(row=2, column=1, padx=(0, 160), pady=(60, 0))

        save_profile_button = customtkinter.CTkButton(self.custom_frame,fg_color="green", width=100, text="Save Details", command=self.save_details)
        save_profile_button.grid(row=2, column=1, padx=(220, 0), pady=(60, 0))

    def dev_settings(self):
        self.custom_frame = customtkinter.CTkFrame(self, height=500)
        self.custom_frame.grid(row=0, column=1, padx=(30, 30), pady=(30, 30), sticky="nsew")

        # Move appearance mode widgets to custom frame
        self.appearance_mode_label.grid_forget()
        self.appearance_mode_optionemenu.grid_forget()

        self.appearance_mode_label = customtkinter.CTkLabel(self.custom_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=0, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.custom_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=1, column=0, padx=20, pady=(10, 10))
        self.appearance_mode_optionemenu.set("Light")

        # Add Camera option menu
        self.camera_label = customtkinter.CTkLabel(self.custom_frame, text="Camera:", anchor="w")
        self.camera_label.grid(row=2, column=0, padx=20, pady=(10, 0))
        self.camera_optionemenu = customtkinter.CTkOptionMenu(self.custom_frame, values=["0", "1", "Enter your input","T@M!M"],
                                                              command=self.set_ipcam)
        self.camera_optionemenu.grid(row=3, column=0, padx=20, pady=(10, 10))
        self.camera_optionemenu.set("0")

  

    def set_ipcam(self, value):
        if value == "Enter your input":
            # Use the last used value as the default input
            default_value = self.ipcam if hasattr(self, 'ipcam') else ""
            self.ipcam = simpledialog.askstring("Enter IP Camera URL", "Please enter the IP camera URL:", initialvalue=default_value)
        else:
            self.ipcam = value

        # Check if the input is an integer
        try:
            self.ipcam = int(self.ipcam)
            print(f"IP Camera set to integer: {self.ipcam}")
        except ValueError:
            print(f"IP Camera set to string: {self.ipcam}")

       


    
            
    def capture_snapshot(self, num_images=20):
        name = self.name_entry.get()
        if not name:
            print("Name entry is empty!")
            return
        
        cap = cv2.VideoCapture(self.ipcam)
        count = 0
        folder_path = os.path.join("KnownFaces", name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow("Capture Faces", frame)
            key = cv2.waitKey(1)
            if key == ord('c') and count < num_images:
                face_path = os.path.join(folder_path, f"{name}_{count}.jpg")
                cv2.imwrite(face_path, frame)
                count += 1
                print(f"Captured {face_path}")
            
            if count >= num_images or key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()




#train

    def ensure_directories(self):
        known_faces_dir = "KnownFaces"
        unknown_faces_dir = "UnknownFaces"
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            print(f"Created directory: {known_faces_dir}")
        if not os.path.exists(unknown_faces_dir):
            os.makedirs(unknown_faces_dir)
            print(f"Created directory: {unknown_faces_dir}")

    def extract_embeddings(self, img, face_rect):
        """
        Extracts 128-dimensional face embeddings using dlib.
        """
        shape = self.predictor(img, face_rect)
        face_descriptor = self.face_rec_model.compute_face_descriptor(img, shape)
        return np.array(face_descriptor)

    def save_details(self):
        self.ensure_directories()
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
                        faces = self.detector(image, 1)
                        if faces:
                            try:
                                face_rect = faces[0]  # Use the first detected face
                                embedding = self.extract_embeddings(image, face_rect)
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
                faces = self.detector(image, 1)
                if faces:
                    try:
                        face_rect = faces[0]  # Use the first detected face
                        embedding = self.extract_embeddings(image, face_rect)
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
    app = App()
    app.mainloop()
