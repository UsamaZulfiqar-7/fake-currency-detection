from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

try:
    model = load_model("model/cnn_model.h5")
except Exception as e:
    print(f"[WARNING] Failed to load model: {e}\nPlease wait for the background training to complete.")
    model = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    img = cv2.resize(img, (256, 256))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# Routes
# =========================

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return "Model is currently training in the background. Please try again in a few minutes.", 503

    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    if file.filename == '':
        return "No file selected", 400
    if not allowed_file(file.filename):
        return "Invalid file type. Upload PNG or JPG images only.", 400

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filename = secure_filename(file.filename)
    if filename == '':
        return "Invalid file name", 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    image_data = prepare_image(filepath)
    prediction_score = model.predict(image_data)[0][0]
    prediction = 0 if prediction_score < 0.5 else 1
    result = "REAL CURRENCY ✅" if prediction == 0 else "FAKE CURRENCY ❌"

    filename_lower = filename.lower()
    if "5000" in filename_lower:
        note_type = "5000 PKR"
    elif "1000" in filename_lower:
        note_type = "1000 PKR"
    elif "500" in filename_lower:
        note_type = "500 PKR"
    elif "100" in filename_lower:
        note_type = "100 PKR"
    elif "50" in filename_lower:
        note_type = "50 PKR"
    else:
        note_type = "Unknown Note"

    security_status = "Authentic ✅" if prediction == 0 else "Suspicious ❌"

    return render_template(
        'index.html',
        result=result,
        image="uploads/" + filename,
        note_type=note_type,
        security_status=security_status
    )

if __name__ == '__main__':

    app.run(debug=True)