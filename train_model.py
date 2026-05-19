import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Feature Extraction Function
def extract_features(image_path):
    img = cv2.imread(image_path, 0)              # grayscale
    img = cv2.resize(img, (256, 256))
    edges = cv2.Canny(img, 100, 200)
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return np.hstack((edges.flatten(), hist))

X = []
y = []

dataset_path = "static/dataset/Real and Fake Currency Dataset"

# Load Real Notes
real_folder = os.path.join(dataset_path, "Real Notes")
for file in os.listdir(real_folder):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        X.append(extract_features(os.path.join(real_folder, file)))
        y.append(0)

# Load Fake Notes
fake_folder = os.path.join(dataset_path, "Fake Notes")
for file in os.listdir(fake_folder):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        X.append(extract_features(os.path.join(fake_folder, file)))
        y.append(1)

X = np.array(X)
y = np.array(y)

print("Dataset Loaded:", len(X), "images")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)
print("Model Trained Successfully")

# Accuracy Check
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save Model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/svm_model.pkl")
print("Model Saved Successfully at model/svm_model.pkl")
