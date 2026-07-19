import os
import json

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'my_custom_data')
TRAINER_OUT = os.path.join(BASE_DIR, 'trainer.yml')
LABEL_MAP_OUT = os.path.join(BASE_DIR, 'label_map.json')
FACE_SIZE = (200, 200)  # normalize all crops to the same size before training
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = []
labels = []
label_map = {}
current_id = 0

if not os.path.isdir(DATASET_PATH):
    raise SystemExit(f"Dataset folder not found: {DATASET_PATH}")

# NOTE: this now only scans the immediate subfolders of DATASET_PATH
# (one folder per person), instead of os.walk()'s full recursive descent.
# os.walk would also process any nested subfolders inside a person's
# folder as if they were separate people, silently mislabeling data.
person_names = sorted(
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
)

if not person_names:
    raise SystemExit(f"No person subfolders found in {DATASET_PATH}")

for person_name in person_names:
    person_path = os.path.join(DATASET_PATH, person_name)
    image_files = [f for f in os.listdir(person_path) if f.lower().endswith(IMAGE_EXTENSIONS)]

    if not image_files:
        print(f"  [skip] '{person_name}' has no image files")
        continue

    label_map[person_name] = current_id
    faces_found_for_person = 0

    for image_name in image_files:
        image_path = os.path.join(person_path, image_name)
        try:
            image = Image.open(image_path).convert('L')
        except (UnidentifiedImageError, OSError) as e:
            print(f"  [warn] could not read '{image_path}': {e}")
            continue

        np_image = np.array(image, 'uint8')
        detected_faces = face_cascade.detectMultiScale(
            np_image, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in detected_faces:
            face_region = np_image[y:y + h, x:x + w]
            face_region = cv2.resize(face_region, FACE_SIZE)
            faces.append(face_region)
            labels.append(current_id)
            faces_found_for_person += 1

    print(f"  '{person_name}' -> id {current_id}: {faces_found_for_person} face(s) from {len(image_files)} image(s)")
    if faces_found_for_person == 0:
        print(f"  [warn] no faces detected for '{person_name}' — this person will never be recognized")

    current_id += 1

if not faces:
    raise SystemExit("No faces were detected in the entire dataset. Nothing to train on.")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
face_recognizer.write(TRAINER_OUT)

with open(LABEL_MAP_OUT, 'w') as f:
    json.dump(label_map, f, indent=2)

print(f"\nModel trained on {len(faces)} face samples across {len(label_map)} people.")
print(f"Saved model to '{TRAINER_OUT}'")
print(f"Saved label map to '{LABEL_MAP_OUT}'")