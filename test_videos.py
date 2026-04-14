import cv2
import os

files = [
    "exam_invigilation_ai/data/videos/front/front.mp4",
    "exam_invigilation_ai/data/videos/back/back.mp4",
    "data/videos/front/front.mp4",
    "data/videos/back/back.mp4"
]

print("Checking video files...")
for f in files:
    exists = os.path.exists(f)
    print(f"File: {f} | Exists: {exists}")
    if exists:
        cap = cv2.VideoCapture(f)
        if cap.isOpened():
            ret, frame = cap.read()
            print(f"  -> Opened: YES | Read Frame: {ret} | Shape: {frame.shape if ret else 'N/A'}")
            cap.release()
        else:
            print(f"  -> Opened: NO")
