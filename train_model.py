import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

IMG_HEIGHT = 128
IMG_WIDTH = 128

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    # MobileNetV2 expects inputs in the range [-1, 1]
    img = img.astype("float32") / 127.5 - 1.0
    return img

def load_dataset(dataset_path):
    images = []
    labels = []
    categories = [("Real Notes", 0), ("Fake Notes", 1)]

    for folder_name, label in categories:
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Dataset folder not found: {folder_path}")

        count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, filename)
                try:
                    images.append(load_image(image_path))
                    labels.append(label)
                    count += 1
                except ValueError as e:
                    print(f"Skipping corrupted image: {e}")

        print(f"Loaded {count} images from '{folder_name}' (label={label})")

    if not images:
        raise ValueError("No images found in the dataset path.")

    return np.array(images), np.array(labels)

def build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False # Freeze base model for fast training

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    dataset_path = os.path.join("static", "dataset", "Real and Fake Currency Dataset")
    X, y = load_dataset(dataset_path)

    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split = int(num_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nDataset loaded: {num_samples} images")
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    model = build_model()
    
    print("Training model...")
    model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=6,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # ---- Per-class accuracy ----
    predictions = (model.predict(X_test) > 0.5).astype(int).flatten()
    real_mask = y_test == 0
    fake_mask = y_test == 1
    real_acc = np.mean(predictions[real_mask] == 0) if np.any(real_mask) else 0
    fake_acc = np.mean(predictions[fake_mask] == 1) if np.any(fake_mask) else 0
    print(f"Real Note Accuracy: {real_acc:.4f}")
    print(f"Fake Note Accuracy: {fake_acc:.4f}")

    os.makedirs("model", exist_ok=True)
    model.save("model/cnn_model.h5")
    print("\nModel saved successfully at model/cnn_model.h5")