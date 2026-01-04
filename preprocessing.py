import cv2
import numpy as np

def preprocess_image(image):
    """
    Input: BGR image (OpenCV format)
    Output: Preprocessed image
    Steps: Grayscale, noise removal, contrast enhancement, background subtraction
    """
    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Noise Removal (Gaussian Blur)
    denoised = cv2.GaussianBlur(gray, (5,5), 0)
    
    # 3. Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 4. Background Subtraction
    # Using simple thresholding as example
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    
    return foreground
