import os
import cv2
import numpy as np
from PIL import Image
import json

# Path to the main dataset folder
main_dataset_path = 'dataset/my_custom_data'

# Create the face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prepare data for training
faces = []
labels = []
label_map = {}
current_id = 0

# Walk through all subfolders in the main dataset path
for root, dirs, files in os.walk(main_dataset_path):
    # This loop gets all the person's folders
    for person_name in dirs:
        person_path = os.path.join(root, person_name)
        
        # We need to make sure we're at a folder with images, not a parent folder like 'lfw-deepfunneled'
        if not any(fname.endswith(('.jpg', '.png')) for fname in os.listdir(person_path)):
            continue

        label_map[person_name] = current_id
        
        # Loop through each image file in the person's folder
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            
            # Read the image and convert to grayscale
            image = Image.open(image_path).convert('L')
            np_image = np.array(image, 'uint8')

            # Detect faces in the image
            detected_faces = face_cascade.detectMultiScale(np_image, 1.3, 5)

            # Extract the face region
            for (x, y, w, h) in detected_faces:
                face_region = np_image[y:y+h, x:x+w]
                faces.append(face_region)
                labels.append(current_id)
        
        current_id += 1

# Create the LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer
face_recognizer.train(faces, np.array(labels))

# Save the trained model to a file
face_recognizer.write('trainer.yml')

# Save the label map to a JSON file
with open('label_map.json', 'w') as f:
    json.dump(label_map, f)

print(f"Model trained successfully and saved to 'trainer.yml'")
print(f"Label map saved to 'label_map.json'")