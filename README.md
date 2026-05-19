# AI-Powered Fake Currency Detection Dashboard

A modern, responsive web application for classifying banknotes as **Authentic (Real)** or **Suspicious (Fake)**. Built using a high-performance Python Flask backend, OpenCV for computer vision feature extraction, and Scikit-Learn for Support Vector Machine (SVM) classification.

---

## 🚀 Key Features

* **Interactive Glassmorphism Dashboard**: Sleek and premium dark/light mode dashboard with floating currency particles and vector scanner animations.
* **Drag-and-Drop Uploader**: Seamless client-side drag-and-drop file interface with real-time image preview.
* **Visual Scanner Simulation**: Synchronous loading bar animation synchronized with an optional auditory indicator upon scanning.
* **Classical Machine Learning Pipeline**: Combines Canny edge detection and normalized grayscale histograms to construct a classification feature space, processed by a Linear SVM.

---

## 🛠️ Tech Stack

* **Backend**: Flask (Python)
* **Computer Vision**: OpenCV (`opencv-python`)
* **Machine Learning**: Scikit-Learn (`scikit-learn`), NumPy, Joblib
* **Frontend**: HTML5, Vanilla CSS3 (Custom Glassmorphism), ES6 JavaScript

---

## 📂 Project Structure

```text
fake-currency-detection/
├── model/                     # Serialized machine learning models (Ignored in Git)
│   └── svm_model.pkl          # Trained Linear SVM model file (~654 MB)
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
1. **Grayscale Conversion**: The input image is converted to grayscale to isolate intensity patterns and remove color variance.
2. **Resizing**: Standardized to $256 \times 256$ pixels to maintain feature shape parity.
3. **Canny Edge Detection**: Identifies fine structural boundaries, security thread patterns, and printing details. The resulting edge-map is flattened into a 65,536-dimensional vector.
4. **Normalized Histogram**: Computes a 256-bin grayscale color histogram to capture the global distribution of light.
5. **Concatenation**: Concatenates both arrays into a **65,792-dimensional** single Numpy feature vector.

### 2. SVM Classification
The feature vector is processed by a pre-trained **Linear Support Vector Classifier (SVC)** to determine authenticity:
* **Label `0`**: Authentic Banknote ✅
* **Label `1`**: Suspicious / Fake Banknote ❌

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
Due to GitHub's size limitations, the `svm_model.pkl` and raw dataset files are ignored in Git. To train the model locally:
1. Ensure your training dataset is placed under `static/dataset/Real and Fake Currency Dataset/` (divided into `Real Notes` and `Fake Notes` subfolders).
2. Execute the training script:
   ```bash
   python train_model.py
   ```
   This will train the model, report test split accuracy, and save the serialized model as `model/svm_model.pkl`.

### 4. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000` to start scanning banknotes.

---

## 🔒 Security & Performance Guidelines (Audit Findings)

During a deep engineering audit, the following recommendations were identified for production deployment:
* **Upload Sanitization**: In production, integrate `werkzeug.utils.secure_filename` to prevent path-traversal attacks and whitelist extensions (`.jpg`, `.png`) to block arbitrary script execution.
* **Model Footprint**: The current SVM is 654MB due to the raw high-dimensional Canny edge space. A planned upgrade involves applying PCA or transitioning to a lightweight Convolutional Neural Network (CNN) such as MobileNetV3 (~10-15MB footprint) to achieve spatial invariance and massive RAM savings.
* **Persistence**: Scan logging and metric stats (scans, counts, and accuracy) can be dynamically stored using a lightweight SQLite database instead of hardcoded landing page variables.
