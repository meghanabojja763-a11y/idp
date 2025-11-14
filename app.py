import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from scipy.spatial import distance as dist
import time

# --------------------------
# 1. FIREBASE SETUP
# --------------------------
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


def mark_attendance(employee_id):
    """Save attendance in Firebase"""
    db.collection("attendance").add({
        "employee_id": employee_id,
        "timestamp": datetime.utcnow()
    })


# --------------------------
# 2. MODEL (ALEXNET)
# --------------------------
class AlexNetFace(nn.Module):
    def __init__(self, num_classes=50):
        super(AlexNetFace, self).__init__()
        self.model = models.alexnet(weights='IMAGENET1K_V1')
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)


# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AlexNetFace(num_classes=50)
model.load_state_dict(torch.load("alexnet_face_model.pth", map_location=device))
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# --------------------------
# 3. LIVENESS DETECTION
# --------------------------
def eye_aspect_ratio(eye):
    """Calculate eye aspect ratio for blink detection"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def detect_blink(landmarks):
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0

    return avg_ear < 0.23  # threshold


def fake_liveness_detector(frame):
    """Simplified liveness (blink detection + motion)"""

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Eye detection using Haarcascade
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return False, None

    (x, y, w, h) = faces[0]
    face = frame[y:y+h, x:x+w]

    # Use dummy landmarks for demonstration
    landmarks = {
        "left_eye": [(10, 20), (15, 18), (20, 20), (10, 24), (15, 25), (20, 24)],
        "right_eye": [(60, 20), (65, 18), (70, 20), (60, 24), (65, 25), (70, 24)]
    }

    blink = detect_blink(landmarks)

    return blink, face


# --------------------------
# 4. FACE RECOGNITION
# --------------------------
def predict_employee(frame):
    face_tensor = transform(frame).unsqueeze(0)

    with torch.no_grad():
        outputs = model(face_tensor)
        _, predicted = torch.max(outputs.data, 1)
        employee_id = int(predicted.item())

    return employee_id


# --------------------------
# STREAMLIT UI
# --------------------------
st.title("🎥 Live Facial Attendance System (AlexNet + Liveness + Firebase)")
st.write("Ensure good lighting and remain in front of the camera.")

start_button = st.button("Start Attendance System")

if start_button:
    stframe = st.empty()

    cap = cv2.VideoCapture(0)
    liveness_confirmed = False
    blink_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Camera not available")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Step 1 — Liveness Check
        blink_detected, face_crop = fake_liveness_detector(frame)

        if blink_detected:
            blink_count += 1

        if blink_count >= 1:  # require at least one blink
            liveness_confirmed = True

        # Step 2 — If live, perform recognition
        if liveness_confirmed and face_crop is not None:
            employee_id = predict_employee(face_crop)

            # Mark attendance
            mark_attendance(employee_id)

            st.success(f"Attendance Recorded for Employee ID: {employee_id}")
            cap.release()
            break

        # Display video stream
        stframe.image(frame_rgb)

        # Timeout after 20 seconds
        if time.time() - start_time > 20:
            st.error("Liveness check failed. Try again.")
            cap.release()
            break

    st.write("Done.")
