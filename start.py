"""
webcam_emotion_recorder.py

Capture webcam video, extract face crops at regular intervals, save them with timestamps,
and send them to DeepFace to predict emotion (face hidden).

Requirements:
    pip install opencv-python deepface numpy
"""

import os
import time
from datetime import datetime
import cv2
from deepface import DeepFace
import numpy as np

# ---------- CONFIG ----------
FRAME_SAVE_DIR = "captured_faces"
INTERVAL_SECONDS = 2.0
WEBCAM_INDEX = 0
ENFORCE_DETECTION = False
DETECTOR_BACKEND = "opencv"
# ----------------------------

os.makedirs(FRAME_SAVE_DIR, exist_ok=True)

haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)
if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade. Check OpenCV installation.")

cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing WEBCAM_INDEX.")

# Create a tiny hidden OpenCV window (for keyboard input)
cv2.namedWindow("EmotionTracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("EmotionTracker", 1, 1)

last_saved = 0.0
sad_start = None  # to store when sadness begins

print("Starting webcam. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam — exiting.")
            break

        # Dummy display for keyboard focus (invisible 1x1 black window)
        cv2.imshow("EmotionTracker", np.zeros((1, 1, 3), dtype=np.uint8))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        now = time.time()
        if now - last_saved >= INTERVAL_SECONDS and len(faces) > 0:
            (x, y, w, h) = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]

            pad = int(0.25 * w)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                last_saved = now
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = os.path.join(FRAME_SAVE_DIR, f"face_{timestamp}.jpg")
            cv2.imwrite(fname, face_crop)

            try:
                result = DeepFace.analyze(
                    img_path=face_crop,
                    actions=['emotion'],
                    enforce_detection=ENFORCE_DETECTION,
                    detector_backend=DETECTOR_BACKEND
                )
                emotions = result[0]['emotion']
                emotion_detected = max(emotions, key=emotions.get)
            except Exception as e:
                print("DeepFace analysis error:", e)
                emotion_detected = "unknown"

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Emotion: {emotion_detected}")

            # ---- Sadness tracking logic ----
            if emotion_detected.lower() == "sad":
                if sad_start is None:
                    sad_start = datetime.now()
                    print(f"😔 Sadness started at {sad_start.strftime('%H:%M:%S')}")
            else:
                if sad_start is not None:
                    sad_end = datetime.now()
                    duration = (sad_end - sad_start).total_seconds()
                    print(f"✅ Sadness ended at {sad_end.strftime('%H:%M:%S')} | Duration: {duration:.2f} sec")
                    sad_start = None

                    # Pause tracking until key pressed
                    print("⏸ Tracking paused. Press any key to resume...")
                    cv2.imshow("EmotionTracker", np.zeros((1, 1, 3), dtype=np.uint8))
                    cv2.waitKey(0)  # Wait indefinitely
                    print("▶️ Tracking resumed...")

            last_saved = now

        # Quit if 'q' or 'Esc' pressed
        key = cv2.waitKey(1)
        if key != -1:
            key = key & 0xFF
            if key == ord('q') or key == 27:
                print("Quitting...")
                break

except KeyboardInterrupt:
    print("Interrupted by user — exiting.")

finally:
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(100)
    print("Finished. Saved frames are in:", os.path.abspath(FRAME_SAVE_DIR))
