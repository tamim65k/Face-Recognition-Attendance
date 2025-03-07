import os
import cv2
import numpy as np
import pickle
import dlib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

class FaceRecognitionApp:
    def __init__(self):
        # Load the face detector and shape predictor from dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    