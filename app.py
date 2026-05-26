from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")

model = load_model("model/cnn_model.h5")


def allowed_file(filename):
    return True


def prepare_image(image_path):
    from PIL import Image
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = img.resize((128, 128))
        img = np.array(img)
        # MobileNetV2 expects inputs in the range [-1, 1]
        img = img.astype("float32") / 127.5 - 1.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        raise ValueError(f"Unable to read image: {image_path}. Details: {e}")

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

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filename = secure_filename(file.filename)
    if filename == '':
        return "Invalid file name", 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        image_data = prepare_image(filepath)
    except Exception as e:
        return f"Error processing image: {str(e)}. The image file might be corrupted, 0 bytes, or an unsupported format.", 400

    prediction_score = model.predict(image_data, verbose=0)[0][0]
    prediction = 0 if prediction_score < 0.5 else 1
    result = "REAL CURRENCY ✅" if prediction == 0 else "FAKE CURRENCY ❌"

    denomination = request.form.get("denomination", "")
    
    if denomination == "5000":
        note_type = "5000 PKR"
        ref_img = "https://placehold.co/400x200/111827/00ffff?text=Real+5000+PKR+Note"
    elif denomination == "1000":
        note_type = "1000 PKR"
        ref_img = "https://placehold.co/400x200/111827/00ffff?text=Real+1000+PKR+Note"
    elif denomination == "500":
        note_type = "500 PKR"
        ref_img = "https://placehold.co/400x200/111827/00ffff?text=Real+500+PKR+Note"
    elif denomination == "100":
        note_type = "100 PKR"
        ref_img = "https://placehold.co/400x200/111827/00ffff?text=Real+100+PKR+Note"
    elif denomination == "50":
        note_type = "50 PKR"
        ref_img = "https://placehold.co/400x200/111827/00ffff?text=Real+50+PKR+Note"
    else:
        note_type = "Unknown Note"
        ref_img = "https://placehold.co/400x200/111827/00ffff?text=Unknown+Note"

    security_status = "Authentic ✅" if prediction == 0 else "Suspicious ❌"

    return render_template(
        'index.html',
        result=result,
        image="uploads/" + filename,
        note_type=note_type,
        security_status=security_status,
        ref_img=ref_img
    )

if __name__ == '__main__':

    app.run(debug=True)