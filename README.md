# AI-Powered Fake Currency Detection Dashboard

A modern, responsive web application for classifying banknotes as **Authentic (Real)** or **Suspicious (Fake)**. Built using a Python Flask backend, OpenCV for preprocessing, and a Convolutional Neural Network (CNN) using TensorFlow/Keras.

---

## 🚀 Key Features

* **Interactive Glassmorphism Dashboard**: Sleek and premium dark/light mode dashboard with floating currency particles and vector scanner animations.
* **Drag-and-Drop Uploader**: Seamless client-side drag-and-drop file interface with real-time image preview.
* **Visual Scanner Simulation**: Synchronous loading bar animation synchronized with an optional auditory indicator upon scanning.
* **Convolutional Neural Network**: Uses a CNN-based image classifier for more robust currency authentication.

---

## 🛠️ Tech Stack

* **Backend**: Flask (Python)
* **Computer Vision**: OpenCV (`opencv-python`)
* **Machine Learning**: TensorFlow (`tensorflow`), NumPy
* **Frontend**: HTML5, Vanilla CSS3 (Custom Glassmorphism), ES6 JavaScript

---

## 📂 Project Structure

```text
fake-currency-detection/
├── model/                     # Serialized machine learning models (Ignored in Git)
│   └── cnn_model.h5           # Trained CNN model file
├── static/                    # Public web asset directory
│   ├── dataset/               # Image training dataset (Ignored in Git)
│   │   ├── Real Notes/
│   │   └── Fake Notes/
│   ├── uploads/               # User scans folder (Ignored in Git)
│   └── style.css              # Main glassmorphism stylesheet (Dark & Light theme)
├── templates/                 # Server-side Jinja2 views
│   └── index.html             # Main dashboard template
├── .gitignore                 # Excludes raw images, large models, and cache
├── app.py                     # Primary Flask web server and api route
├── fake_currency_detection.py # Standalone local model evaluator and runner
├── train_model.py             # Model training and serialization script
├── test.jpeg                  # Sample banknote for quick local verification
├── requirements.txt           # Environment package configuration
└── README.md                  # Project documentation
```

---

## ⚙️ How It Works

### 1. Preprocessing & Feature Extraction
Banknote verification is executed via the `extract_features` pipeline:
1. **Grayscale Conversion**: The input image is converted to grayscale to isolate intensity patterns and reduce color noise.
2. **Resizing**: Standardized to 256×256 pixels to create consistent CNN input.
3. **Normalization**: Pixel values are scaled to the [0, 1] range before inference.

### 2. CNN Classification
The preprocessed image is passed through a Convolutional Neural Network (CNN) trained with TensorFlow/Keras:
* **Output near `0`**: Authentic Banknote ✅
* **Output near `1`**: Suspicious / Fake Banknote ❌

---

## 💻 Quick Setup & Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/UsamaZulfiqar-7/fake-currency-detection.git
cd fake-currency-detection
```

### 2. Install Dependencies
It is highly recommended to use a Python virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Place Dataset & Train the Model
Due to GitHub's size limitations, the trained model and raw dataset files are ignored in Git. To train the model locally:
1. Ensure your training dataset is placed under `static/dataset/Real and Fake Currency Dataset/` (divided into `Real Notes` and `Fake Notes` subfolders).
2. Execute the training script:
   ```bash
   python train_model.py
   ```
   This will train the CNN, report test accuracy, and save the serialized model as `model/cnn_model.h5`.

### 4. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000` to start scanning banknotes.

---

## 🔒 Security & Performance Guidelines (Audit Findings)

During a deep engineering audit, the following recommendations were identified for production deployment:
* **Upload Sanitization**: In production, integrate `werkzeug.utils.secure_filename` to prevent path-traversal attacks and whitelist extensions (`.jpg`, `.png`) to block arbitrary script execution.
* **Model Footprint**: The current implementation uses a CNN saved as `model/cnn_model.h5`. For production, consider a more compact architecture or model quantization to reduce disk usage and inference latency.
* **Persistence**: Scan logging and metric stats (scans, counts, and accuracy) can be dynamically stored using a lightweight SQLite database instead of hardcoded landing page variables.
