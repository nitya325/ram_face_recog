import cv2
import numpy as np
import json

# Load the trained face recognizer and the Haar Cascade face detector
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainer.yml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the label map
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

# Invert the label map for easy lookup (from ID to name)
reverse_label_map = {int(v): k for k, v in label_map.items()}

# Access the webcam (0 is typically the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting real-time face recognition... Press 'q' to exit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the face region for recognition
        face_region = gray_frame[y:y + h, x:x + w]

        # Predict the person's ID and confidence level
        person_id, confidence = face_recognizer.predict(face_region)

        # Look up the name from the label map
        person_name = reverse_label_map.get(person_id, "Unknown")

        # Display the name and confidence on the frame
        if confidence < 100:  # A lower confidence score means a better match
            text = f"{person_name} (Confidence: {round(confidence, 2)})"
        else:
            text = "Unknown"

        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the final frame
    cv2.imshow('Real-time Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close all windows
cap.release()
cv2.destroyAllWindows()