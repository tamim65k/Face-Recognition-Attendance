# Face Recognition Attendance System

## Overview

This project is a Face Recognition Attendance System that leverages face recognition technology to manage and track attendance efficiently. The system uses OpenCV and Dlib for face detection and recognition, and integrates a liveness detection mechanism to prevent spoofing. This README provides a comprehensive guide on how to set up, run, and use the system.

## Features

- **Face Detection and Recognition**: Uses Dlib’s face recognition model.
- **Liveness Detection**: Implements anti-spoofing measures to ensure security.
- **Real-time Processing**: Captures and processes video frames from the webcam.
- **Attendance Logging**: Records attendance in a CSV file with timestamp.
- **User-Friendly Interface**: Built using customtkinter for a modern GUI.

## Requirements

- Python 3.x
- Libraries: OpenCV, Dlib, NumPy, TensorFlow (or PyTorch for models in .pth format), customtkinter

## Installation

1. **Clone the Repository**:

    git clone https://github.com/yourusername/face-recognition-attendance-system.git
    cd face-recognition-attendance-system


2. **Install Dependencies**:

    pip install opencv-python dlib numpy tensorflow customtkinter torch torchvision


3. **Download Models**:
   - Place the required face recognition model files (`shape_predictor_68_face_landmarks.dat` and `dlib_face_recognition_resnet_model_v1.dat`) in the project directory.
   - Ensure your anti-spoofing models (`model1.pth` and `model2.pth`) are in the project directory.

## Usage

1. **Run the Attendance System**:

    python gui.py


2. **Real-time Liveness Detection**:
   The system will start the webcam and display real-time liveness detection results.

3. **Registering New Users**:
   - Capture images of new users and save them in the `KnownFaces` directory under a subdirectory named after the user.
   - Update the `labels.csv` file accordingly.

## Project Structure


face-recognition-attendance-system/
│
├── KnownFaces/                   # Directory containing known user face images
│   ├── user1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── user2/
│       ├── image1.jpg
│       └── image2.jpg
│
├── UnknownFaces/                 # Directory for storing unknown user face images
│   ├── unknown1.jpg
│   └── unknown2.jpg
│
├── models/                       # Directory for storing models
│   ├── shape_predictor_68_face_landmarks.dat
│   ├── dlib_face_recognition_resnet_model_v1.dat
│   ├── model1.pth
│   └── model2.pth
│
├── liveness_detection_model.h5   # Trained liveness detection model (if applicable)
│
├── attendance.csv                # CSV file for logging attendance
│
├── labels.csv                    # CSV file for mapping image files to labels
│
├── gui.py                       # Main script for running the system
│
└── README.md                     # This README file


## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

- Dlib for face detection and recognition.
- OpenCV for computer vision tasks.
- TensorFlow/PyTorch for machine learning models.

