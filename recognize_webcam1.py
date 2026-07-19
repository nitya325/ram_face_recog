
tally · current session
38%
≈62 sonnet msgs
resets in 4h 38m
weekly · all models
2% · Wed 5:29 PM




































Recognize webcam1 · PY
"""
Desktop-only real-time face recognition via a local webcam window.
 
This script requires a physical camera and a display (cv2.imshow opens a
native GUI window) — it is NOT deployable to a cloud server. For a
browser/cloud-friendly version of this same functionality, see app.py +
templates/index.html, which streams frames from the browser's webcam via
getUserMedia instead of opening a device handle directly.
"""
import os
import sys
import json
 
import cv2
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINER_PATH = os.path.join(BASE_DIR, 'trainer.yml')
LABEL_MAP_PATH = os.path.join(BASE_DIR, 'label_map.json')
 
# Kept identical to app.py's threshold so behavior matches between the two
# entry points. LBPH confidence is a distance: lower = better match.
CONFIDENCE_THRESHOLD = 70.0
MIN_FACE_SIZE = (60, 60)
 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
try:
    face_recognizer.read(TRAINER_PATH)
except cv2.error:
    print(f"Error: could not load '{TRAINER_PATH}'. Run train_model.py first.")
    sys.exit(1)
 
try:
    with open(LABEL_MAP_PATH, 'r') as f:
        reverse_label_map = {int(v): k for k, v in json.load(f).items()}
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error: could not load '{LABEL_MAP_PATH}': {e}")
    sys.exit(1)
 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)
 
print("Starting real-time face recognition... Press 'q' to exit.")
 
while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: failed to read a frame from the webcam.")
        break
 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=MIN_FACE_SIZE
    )
 
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_region = gray_frame[y:y + h, x:x + w]
 
        try:
            person_id, confidence = face_recognizer.predict(face_region)
        except cv2.error:
            continue
 
        if confidence < CONFIDENCE_THRESHOLD:
            name = reverse_label_map.get(person_id, "Unknown")
            text = f"{name} (Confidence: {round(confidence, 2)})"
        else:
            text = "Unknown"
 
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 
    cv2.imshow('Real-time Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
 
