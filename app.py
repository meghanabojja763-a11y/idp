import streamlit as st
import cv2
from preprocessing import preprocess_image
import numpy as np

st.title("Live Photo Preprocessing App")

# Upload image or capture from webcam
option = st.radio("Choose input method:", ["Upload Image", "Webcam Capture"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        processed = preprocess_image(img)
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Preprocessed Image")
        
elif option == "Webcam Capture":
    st.text("Click 'Capture' to take photo from webcam")
    if st.button("Capture"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            processed = preprocess_image(frame)
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Preprocessed Image")
