# Real-Time Face Recognition System

A real-time face recognition app built with **Python, OpenCV (LBPH), and Flask**. It supports two usage modes: a **browser-based web app** (webcam capture happens client-side, inference happens server-side — deployable to the cloud) and a **local desktop script** (runs an OpenCV window directly against your machine's webcam).

<img width="922" height="867" alt="Screenshot 2026-07-19 172532" src="https://github.com/user-attachments/assets/dd828565-1a3b-49c2-9cdd-4a58bb7f6861" />


**Live demo:** `<add your deployed URL here once live>`

---

## How it works

1. **Face detection** — a Haar Cascade classifier (`haarcascade_frontalface_default.xml`, shipped with OpenCV) locates face bounding boxes in a grayscale frame.
2. **Face recognition** — each detected face crop is passed to an **LBPH (Local Binary Pattern Histogram) recognizer**, which was previously trained on labeled example photos of known people. LBPH returns a predicted person ID and a *distance score* (lower = more confident match).
3. **Thresholding** — if the distance score is above a configured threshold, the prediction is discarded and the face is labeled "Unknown" rather than reporting a low-confidence guess as fact.

### Two front ends, one recognition core

| | `app.py` (web) | `recognize_webcam1.py` (desktop) |
|---|---|---|
| Where the webcam is accessed | In the **browser**, via `getUserMedia` | Directly on the machine, via `cv2.VideoCapture(0)` |
| Where inference runs | Server-side, per uploaded frame | Locally, in the same process |
| Deployable to the cloud? | **Yes** | No — a cloud server has no physical camera |
| Output | JSON (bounding boxes + names) rendered in HTML/CSS overlays | An OpenCV GUI window |

This split matters: a naive "just run OpenCV's webcam loop on a server" approach is a dead end for deployment, since remote servers don't have a camera. The web app instead captures frames **in the user's browser** and sends individual JPEG frames to the backend for inference — the correct architecture for any cloud-hosted webcam app.

---

## Project structure

```
.
├── app.py                   # Flask web server (deployable)
├── recognize_webcam1.py     # Local desktop script (not deployable)
├── train_model.py           # Trains the LBPH model on dataset/my_custom_data
├── templates/
│   └── index.html           # Browser UI + webcam capture + overlay rendering
├── trainer.yml              # Trained LBPH model (binary, committed for deployment)
├── label_map.json           # Maps numeric label IDs -> person names
├── requirements.txt
├── Procfile                 # For gunicorn-based deployment (Render/Railway/Heroku-style)
└── dataset/                 # Training images (NOT committed — see .gitignore)
    └── my_custom_data/
        └── <person_name>/
            ├── 001.jpg
            └── 002.jpg
```

---

## Features

- Real-time face detection + recognition from a live video feed
- Confidence-thresholded predictions (unknown faces are labeled "Unknown" instead of guessed)
- Browser-based UI with live bounding-box + name overlays, correctly scaled to the displayed video size
- Trainable on your own custom dataset — just add a folder per person
- Clean separation between the deployable web app and the local-only desktop demo
- `/health` endpoint for deployment platform uptime checks

---

## Setup

### Prerequisites
- Python 3.9–3.11
- A webcam (for local use)

### Install

```bash
git clone https://github.com/nitya325/ram_face_recog.git
cd ram_face_recog
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs `opencv-contrib-python-headless`, which is correct for the **web app** (no GUI needed on a server) but does **not** support `cv2.imshow()`. If you want to run the local desktop script (`recognize_webcam1.py`), also install the GUI-capable build locally:

```bash
pip install opencv-contrib-python
```

(Only one of the two `opencv-contrib-python*` variants can be installed at a time — the desktop script is a local dev convenience, not part of the deployed app, so this is a one-line local swap, not something that needs to go in `requirements.txt`.)

### Train the model

Structure your dataset as:

```
dataset/my_custom_data/
├── nitya/
│   ├── 001.jpg
│   └── 002.jpg
└── vanshika/
    ├── 001.jpg
    └── 002.jpg
```

Then run:

```bash
python train_model.py
```

This produces `trainer.yml` and `label_map.json`, which both `app.py` and `recognize_webcam1.py` load at startup. Aim for **15–30+ varied photos per person** (different angles, lighting, expressions) — LBPH is sensitive to sparse or overly-similar training data.

### Run locally

**Web app:**
```bash
python app.py
```
Visit `http://127.0.0.1:5000`.

**Desktop script:**
```bash
python recognize_webcam1.py
```
Press `q` to quit.

---

## Deployment

See the deployment steps below — the short version: this repo is already deployment-ready (`Procfile`, `requirements.txt`, `PORT`/`FLASK_DEBUG` env-var support, headless OpenCV). Push to GitHub, connect the repo on Render (or Railway/Fly.io), and it builds and serves automatically. The camera never touches the server — it stays in the visitor's browser the whole time.

---

## Limitations

- **LBPH is a classical (non-deep-learning) method.** It's fast and lightweight, but noticeably less robust than deep embeddings (e.g., FaceNet/ArcFace) to pose variation, lighting changes, occlusion, and aging — see the study guide for a full trade-off comparison.
- **No liveness detection.** A printed photo or a phone screen held up to the camera can fool it. Not suitable for any actual security/access-control use case as-is.
- **Small, self-collected training set.** Accuracy depends heavily on how many varied photos exist per person; a handful of near-identical selfies will overfit and generalize poorly to new lighting/angles.
- **Single-face-per-frame confidence heuristic on the UI** — the interface highlights the first detected face as the "main" one, which is a display simplification, not a ranking of certainty.
- **Not tested for bias/fairness** across demographics — a known, well-documented weakness of face recognition systems generally, and something a from-scratch personal project like this has not attempted to audit.

---

## Tech stack

Python · OpenCV (Haar Cascade + LBPH) · Flask · NumPy · Pillow · vanilla JS (`getUserMedia`, `Canvas`, `fetch`) · gunicorn![Uploading Screenshot 2026-07-19 172532.png…]()
