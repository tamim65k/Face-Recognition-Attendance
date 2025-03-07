import cv2
import os

def capture_faces(name, num_images=20):
    cap = cv2.VideoCapture(2)
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

if __name__ == "__main__":
    name = input("Enter the name of the person: ")
    capture_faces(name)
