import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

IMG_HEIGHT = 256
IMG_WIDTH = 256


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}")
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)


def load_dataset(dataset_path):
    images = []
    labels = []
    categories = [("Real Notes", 0), ("Fake Notes", 1)]

    for folder_name, label in categories:
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Dataset folder not found: {folder_path}")

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, filename)
                images.append(load_image(image_path))
                labels.append(label)

    if not images:
        raise ValueError("No images found in the dataset path.")

    return np.array(images), np.array(labels)


def build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
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

    print("Dataset loaded:", num_samples, "images")
    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))

    model = build_model()
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=25,
        batch_size=16,
        validation_split=0.15,
        callbacks=[early_stopping],
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    os.makedirs("model", exist_ok=True)
    model.save("model/cnn_model.h5")
    print("Model saved successfully at model/cnn_model.h5")
