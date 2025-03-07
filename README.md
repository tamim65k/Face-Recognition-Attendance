# Face Recognition-Based Attendance System with Liveness Detection


## Project Overview
A real-time face recognition system with advanced **anti-spoofing features** to prevent identity theft via photos/videos. The system uses:
1. Blink detection
2. Head movement analysis
3. Texture verification
4. SVM-based face recognition

---

## Table of Contents
1. [Features](#features)
2. [Working Procedure](#working-procedure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Technologies Used](#technologies-used)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)

---

## Features
✅ **Real-Time Face Detection**  
✅ **Liveness Checks** (Blink Detection + Head Movement + Texture Analysis)  
✅ **Attendance Logging** (CSV files with timestamps)  
✅ **User Registration** via GUI  
✅ **Camera Configuration** (IP/USB camera selection)  
✅ **Dark/Light Mode** interface  

---

## Working Procedure
### 1. Face Detection & Landmark Tracking
- Uses **dlib's frontal face detector** to find faces in frames
- **68-point facial landmarks** are tracked for eye/face analysis

### 2. Liveness Detection
- **Blink Detection**: 
  - Calculates Eye Aspect Ratio (EAR) to detect blinks
  - Requires at least 1 blink during authentication
- **Head Movement**: 
  - Tracks head position changes between frames
- **Texture Analysis**: 
  - Uses Local Binary Patterns (LBP) to verify natural skin texture

### 3. Face Recognition
- **Face Embeddings**: 
  - 128-dimensional vectors generated using dlib's ResNet model
- **SVM Classifier**: 
  - Trained on registered faces to recognize identities
  - Probability threshold of 60%+ required for authentication

### 4. Attendance Tracking
- Logs approved entries in CSV files
- Prevents duplicate entries through real-time tracking

---
## Technologies Used
| **Component**          | **Library/Tool**                                                                 |
|-------------------------|---------------------------------------------------------------------------------|
| **GUI**                | CustomTkinter (Modern Tkinter)                                                 |
| **Face Detection**     | dlib's HOG-based detector                                                      |
| **Landmark Detection** | dlib's shape predictor (68-point facial landmarks)                             |
| **Face Recognition**   | dlib's ResNet model (128-dimensional embeddings)                               |
| **ML Classification**  | SVM (Support Vector Machine) with probability estimates (scikit-learn)         |
| **Utility Functions**  | OpenCV (computer vision), NumPy (numerical computing), SciPy (scientific tools) |

---

## Usage 📸🔍

### 1. **Register New User**  
**Steps to enroll a user:**  
1. Click **`New Registration`** in the sidebar  
2. Enter a **nickname** in the input field  
3. Press **`Capture Snapshot`** to take **20 face images**  
   - Press **`c`** to capture a frame  
   - Press **`q`** to exit early  
4. Click **`Save Details`** to train the model  

### 2. **Take Attendance**  
**Process for real-time tracking:**  
1. Configure camera in **`Dev Settings`**  
   - Select camera index or input custom URL  
2. Click **`Mark Attendance`** to start detection  
3. The system will:  
   - Track faces in real-time  
   - Validate liveness (see requirements below)  
   - Log valid entries in `attendance/YYYY-MM-DD.csv`  
4. Press **`q`** to stop attendance  

### 3. **Liveness Detection Requirements**  
To be marked as **`Real`**, the system requires:  
✅ **At least 1 blink detected**  
✅ **Head movement** (minimum 10px shift)  
✅ **Valid skin texture patterns** (LBP analysis)  
✅ **≥60% recognition probability** (SVM confidence threshold)  

---

💡 **Note**:  
- Ensure `KnownFaces` directory contains enrolled users  
- Pre-trained models (`shape_predictor_68...`, `dlib_face_recognition...`) must be in root folder  
- Attendance logs are timestamped and date-organized  