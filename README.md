# Real-Time Face Recognition App (OpenCV & Flask)

This project implements a robust, real-time face recognition system using Python, the **OpenCV LBPH (Local Binary Pattern Histograms) algorithm**, and a **Flask** web server for demonstration.

The system is designed to be trained on custom datasets and offers two methods for recognition: a browser-based web interface and a standalone desktop application.

---

## Features

* **Model Training:** Uses `train_model.py` to train an LBPH model from local images, generating `trainer.yml` and `label_map.json`.
* **Web Recognition:** The Flask application (`app.py`) provides a live, browser-based video stream interface for recognition.
* **Desktop Recognition:** Includes a standalone script (`recognize_webcam1.py`) for simple, non-web recognition via a local webcam window.
* **Clean Repository:** Uses `.gitignore` to exclude large, auto-generated files like the virtual environment (`venv/`), the dataset (`dataset/`), and the trained model files (`trainer.yml`, `label_map.json`).

---

## Prerequisites

Ensure you have the following installed on your system:

* **Python 3.x**
* **Git**

All necessary Python dependencies will be installed automatically from the `requirements.txt` file.

---

## Setup and Installation

Follow these steps to set up the environment and install all necessary libraries.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nitya325/ram_face_recog.git](https://github.com/nitya325/ram_face_recog.git)
    cd ram_face_recog
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows (PowerShell/CMD/Git Bash):
    ./venv/Scripts/activate
    ```

3.  **Install Dependencies:**
    This installs necessary libraries like `opencv-python` and `Flask`.
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 Usage

The application requires a model to be trained before recognition can occur.

### Step 1: Prepare the Dataset

Before training, you must create your training data. The `train_model.py` script expects the images to be structured inside a subfolder of `dataset/` (e.g., `dataset/my_custom_data`).

The structure should look like this:
dataset/
└── my_custom_data/
├── person_1_name/
│   ├── 001.jpg
│   └── 002.jpg
└── person_2_name/
├── 001.jpg
└── 002.jpg

***Note:*** *The `dataset` folder is ignored by Git and must be created locally.*

### Step 2: Train the Model

Run the training script to generate the necessary recognition files (`trainer.yml` and `label_map.json`).

```bash
python train_model.py
```

Step 3A: Run the Web App (Flask)
Start the Flask application to access the recognition via a browser.

```bash
python app.py
```
Access the app at http://127.0.0.1:5000 in your web browser.

Step 3B: Run the Desktop Webcam App
Run the standalone script for a quick desktop-based recognition using a dedicated OpenCV window.

```bash

python recognize_webcam1.py
```

(Press 'q' on your keyboard to exit the application.)

### Next Steps:

1.  **Edit `README.md`:** Open the `README.md` file in your project folder (`D:\ram_face_recog`) and replace the old "Usage" section with the revised content above.
2.  **Save** the file.
3.  **Push the update** to GitHub:
    ```bash
    git add README.md
    git commit -m "Refactor README usage section for clarity"
    git push
    ```
