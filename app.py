from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import joblib

app = Flask(__name__)

# Load Model
model = joblib.load("model/svm_model.pkl")

# =========================
# Feature Extraction
# =========================

def extract_features(image_path):

    img = cv2.imread(image_path, 0)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    img = cv2.resize(img, (256, 256))

    edges = cv2.Canny(img, 100, 200)

    hist = cv2.calcHist([img], [0], None, [256], [0,256])

    hist = cv2.normalize(hist, hist).flatten()

    return np.hstack((edges.flatten(), hist))

# =========================
# Routes
# =========================

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    if file.filename == '':
        return "No file selected", 400

    # Create Upload Folder
    os.makedirs("static/uploads", exist_ok=True)

    filepath = os.path.join("static/uploads", file.filename)

    file.save(filepath)

    # =========================
    # Prediction
    # =========================

    features = extract_features(filepath)

    prediction = model.predict([features])[0]

    result = "REAL CURRENCY ✅" if prediction == 0 else "FAKE CURRENCY ❌"

    # =========================
    # Detect Note Type
    # =========================

    filename = file.filename.lower()

    if "5000" in filename:

        note_type = "5000 PKR"

    elif "1000" in filename:

        note_type = "1000 PKR"

    elif "500" in filename:

        note_type = "500 PKR"

    elif "100" in filename:

        note_type = "100 PKR"

    elif "50" in filename:

        note_type = "50 PKR"

    else:

        note_type = "Unknown Note"

    # =========================
    # Fake/Real Status
    # =========================

    if prediction == 0:

        security_status = "Authentic ✅"

    else:

        security_status = "Suspicious ❌"

    return render_template(

        'index.html',

        result=result,

        image="uploads/" + file.filename,

        note_type=note_type,

        security_status=security_status

    )

if __name__ == '__main__':

    app.run(debug=True)