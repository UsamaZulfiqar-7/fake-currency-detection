import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ===============================
# Feature Extraction Function
# ===============================
def extract_features(image_path):
    img = cv2.imread(image_path, 0)              # grayscale
    img = cv2.resize(img, (256, 256))           # resize to standard
    edges = cv2.Canny(img, 100, 200)            # edge detection
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return np.hstack((edges.flatten(), hist))

# ===============================
# Dataset Loading
# ===============================
X = []
y = []

dataset_path = "static/dataset/Real and Fake Currency Dataset"

# Nested folders: Real Notes and Fake Notes
real_folder = os.path.join(dataset_path, "Real Notes")
fake_folder = os.path.join(dataset_path, "Fake Notes")

# Load Real Images
for file in os.listdir(real_folder):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(real_folder, file)
        X.append(extract_features(img_path))
        y.append(0)  # Real label

# Load Fake Images
for file in os.listdir(fake_folder):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(fake_folder, file)
        X.append(extract_features(img_path))
        y.append(1)  # Fake label

X = np.array(X)
y = np.array(y)

print("Dataset Loaded Successfully")
print("Total Images:", len(X))

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Train SVM Model
# ===============================
model = SVC(kernel='linear')
model.fit(X_train, y_train)
print("Model Trained Successfully")

# ===============================
# Accuracy Check
# ===============================
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ===============================
# Test Single Image
# ===============================
test_image = "test.jpeg"   # Project folder me rakho
features = extract_features(test_image)
prediction = model.predict([features])

if prediction[0] == 0:
    print("✅ REAL CURRENCY")
else:
    print("❌ FAKE CURRENCY")
