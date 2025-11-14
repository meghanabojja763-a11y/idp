# utils.py
import cv2
import dlib
import numpy as np
from imutils import face_utils

# Initialize dlib detector (load predictor path later)
detector = dlib.get_frontal_face_detector()

def detect_faces(gray_image):
    """Return dlib rectangles for detected faces in a grayscale image."""
    rects = detector(gray_image, 0)
    return rects

def crop_face_from_rect(frame, rect, expand=0.2):
    """
    Given a frame (BGR) and a dlib rect, return a cropped RGB face image (square).
    expand: fraction to expand bounding box around face
    """
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cx = x + w//2
    cy = y + h//2
    side = max(w, h)
    side = int(side * (1 + expand))
    x1 = max(0, cx - side//2)
    y1 = max(0, cy - side//2)
    x2 = min(frame.shape[1], cx + side//2)
    y2 = min(frame.shape[0], cy + side//2)
    crop = frame[y1:y2, x1:x2]
    return crop

def draw_box(frame, rect, label=None):
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    if label:
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
