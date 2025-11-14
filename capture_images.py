# capture_images.py
import os
import cv2
import argparse
from utils import detect_faces, crop_face_from_rect, draw_box

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, help='Person name (folder will be created)')
parser.add_argument('--num', type=int, default=50, help='Number of face images to capture')
parser.add_argument('--out', default='data/recognition', help='Root output folder')
parser.add_argument('--cam', type=int, default=0, help='Camera index (default 0)')
args = parser.parse_args()

person_dir = os.path.join(args.out, args.name)
os.makedirs(person_dir, exist_ok=True)

cap = cv2.VideoCapture(args.cam)
count = len([f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg','.png'))])

print(f"Starting capture for '{args.name}'. Press 'c' to capture, 'q' to quit.")
print(f"Already {count} images in folder. Need {args.num} images total.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detect_faces(gray)
    for rect in rects:
        draw_box(frame, rect)
    cv2.putText(frame, f"Captured: {count}/{args.num}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Capture - press c to save face", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if len(rects) == 0:
            print("No face detected — try again.")
            continue
        # use the first detected face
        face = crop_face_from_rect(frame, rects[0], expand=0.25)
        if face.size == 0:
            print("Empty crop — skip.")
            continue
        fname = os.path.join(person_dir, f"{args.name}_{count+1:03d}.jpg")
        cv2.imwrite(fname, face)
        count += 1
        print("Saved:", fname)
        if count >= args.num:
            print("Captured required images.")
            break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
