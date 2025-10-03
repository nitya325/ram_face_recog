from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import json
import base64

# --- Load the trained models and label map ---
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainer.yml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
with open('label_map.json', 'r') as f:
    reverse_label_map = {int(v): k for k, v in json.load(f).items()}

app = Flask(__name__)

# Parameters
CONFIDENCE_THRESHOLD = 70.0 # Only consider matches with confidence below this value

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize_face():
    # Decode the image from base64
    data = request.json
    image_data_base64 = data['image_data'].split(',')[1]
    image_data_bytes = base64.b64decode(image_data_base64)
    
    np_image = np.frombuffer(image_data_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    results = []
    
    for (x, y, w, h) in faces:
        face_region = gray_image[y:y+h, x:x+w]
        
        person_id, confidence = face_recognizer.predict(face_region)

        # Apply confidence filtering
        name = reverse_label_map.get(person_id, "Unknown")
        if confidence >= CONFIDENCE_THRESHOLD:
            name = "Unknown"
        
        results.append({
            'name': name,
            'confidence': round(confidence, 2),
            'face_location': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        })
    
    # Return the results as a JSON object
    return jsonify({'faces': results})

if __name__ == '__main__':
    app.run(debug=True)