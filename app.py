import os
import sys
import base64
import json

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template

# --- Paths (absolute, so this works regardless of the working directory
#     the process is launched from — important for deployment platforms) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINER_PATH = os.path.join(BASE_DIR, 'trainer.yml')
LABEL_MAP_PATH = os.path.join(BASE_DIR, 'label_map.json')

# --- Parameters ---
# LBPH "confidence" is actually a DISTANCE score: lower = better match.
# 0 = pixel-perfect match, ~100+ = essentially no match. There is no
# universal correct threshold — tune this against your own dataset.
CONFIDENCE_THRESHOLD = 70.0
MIN_FACE_SIZE = (60, 60)  # ignore tiny false-positive detections

app = Flask(__name__)

# --- Load model + label map once at startup, and fail loudly (not with a
#     cryptic stack trace) if the artifacts are missing ---
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("FATAL: Haar cascade failed to load from OpenCV's data path.", file=sys.stderr)
    sys.exit(1)

try:
    face_recognizer.read(TRAINER_PATH)
except cv2.error:
    print(f"FATAL: could not load '{TRAINER_PATH}'. Run train_model.py first.", file=sys.stderr)
    sys.exit(1)

try:
    with open(LABEL_MAP_PATH, 'r') as f:
        reverse_label_map = {int(v): k for k, v in json.load(f).items()}
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"FATAL: could not load '{LABEL_MAP_PATH}': {e}", file=sys.stderr)
    sys.exit(1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    """Simple endpoint deployment platforms can poll to confirm the app is up."""
    return jsonify({'status': 'ok', 'known_people': len(reverse_label_map)})


@app.route('/recognize', methods=['POST'])
def recognize_face():
    data = request.get_json(silent=True)
    if not data or 'image_data' not in data:
        return jsonify({'error': 'Missing image_data'}), 400

    try:
        header, _, b64_payload = data['image_data'].partition(',')
        image_bytes = base64.b64decode(b64_payload or header)
    except (ValueError, TypeError):
        return jsonify({'error': 'Malformed base64 image data'}), 400

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Could not decode image'}), 400

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.3, minNeighbors=5, minSize=MIN_FACE_SIZE
    )

    results = []
    for (x, y, w, h) in faces:
        face_region = gray_image[y:y + h, x:x + w]
        try:
            person_id, confidence = face_recognizer.predict(face_region)
        except cv2.error:
            # Can happen on a degenerate/near-empty crop; skip this detection.
            continue

        name = reverse_label_map.get(person_id, "Unknown")
        if confidence >= CONFIDENCE_THRESHOLD:
            name = "Unknown"

        results.append({
            'name': name,
            'confidence': round(float(confidence), 2),
            'face_location': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        })

    return jsonify({'faces': results})


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    # host='0.0.0.0' is required for cloud platforms (Render/Railway/etc.)
    # to route external traffic to the container.
    app.run(host='0.0.0.0', port=port, debug=debug_mode)