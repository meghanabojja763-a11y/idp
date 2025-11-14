# app.py
import streamlit as st
import av
import cv2
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import time
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from utils import detect_faces, crop_face_from_rect, draw_box

st.set_page_config(page_title="Face Attendance (AlexNet)", layout="centered")

MODEL_PATH = "models/recognition_alexnet.pth"
SHAPE_PATH = "shape_predictor_68_face_landmarks.dat"
ATTENDANCE_FILE = "attendance.csv"

st.title("Face Attendance Portal (AlexNet)")

# Sidebar: load model and settings
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Recognition probability threshold", 0.3, 0.99, 0.6, 0.05)
show_box = st.sidebar.checkbox("Show face box", value=True)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}. Train model first (train_recognition.py).")
    st.stop()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(MODEL_PATH, map_location=device)
classes = ckpt['classes']

# Build alexnet model skeleton and load weights
model = models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, len(classes))
model.load_state_dict(ckpt['model_state'])
model = model.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Attendance store in session state
if "attendance" not in st.session_state:
    if os.path.exists(ATTENDANCE_FILE):
        st.session_state.attendance = pd.read_csv(ATTENDANCE_FILE).to_dict('records')
    else:
        st.session_state.attendance = []

st.write("Model classes:", ", ".join(classes))

# Helper to write attendance to CSV
def mark_attendance(name):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.attendance.append({'name': name, 'timestamp': timestamp})
    df = pd.DataFrame(st.session_state.attendance)
    df.to_csv(ATTENDANCE_FILE, index=False)
    st.success(f"Marked attendance: {name} at {timestamp}")

# Video transformer for streamlit-webrtc
class FaceTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.device = device
        self.preprocess = preprocess
        self.classes = classes
        self.threshold = threshold
        self.last_mark_time = {}  # name -> last timestamp to avoid repeated writes

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        display = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detect_faces(gray)
        if len(rects) == 0:
            # show not matched text
            cv2.putText(display, "No face detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            return av.VideoFrame.from_ndarray(display, format="bgr24")

        # process first face only
        rect = rects[0]
        face = crop_face_from_rect(img, rect, expand=0.25)
        if face is None or face.size == 0:
            return av.VideoFrame.from_ndarray(display, format="bgr24")
        # convert to PIL and preprocess
        pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        inp = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(inp)
            probs = torch.softmax(out, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)
            conf_val = float(conf.cpu().numpy())
            label = self.classes[int(pred.cpu().numpy())]

        if show_box:
            draw_box(display, rect, label=f"{label}:{conf_val:.2f}")

        # Decide match / not matched
        if conf_val >= self.threshold:
            # prevent rapid repeat marking: only mark once every 10 seconds per person
            last = self.last_mark_time.get(label, 0)
            now = time.time()
            if now - last > 10:
                mark_attendance(label)
                self.last_mark_time[label] = now
            cv2.putText(display, f"Matched: {label} ({conf_val:.2f})", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(display, f"Not matched ({conf_val:.2f})", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        return av.VideoFrame.from_ndarray(display, format="bgr24")

# Start streaming
st.sidebar.markdown("---")
st.sidebar.write("Start webcam and mark attendance in real time.")
webrtc_ctx = webrtc_streamer(
    key="face-attendance",
    mode="live",
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_transformer_factory=FaceTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# Show attendance table
st.header("Attendance records")
if len(st.session_state.attendance) == 0:
    st.write("No attendance yet.")
else:
    df = pd.DataFrame(st.session_state.attendance)
    st.dataframe(df)
