import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

print("==============================================")
print("Smart Cauliflower - Image Size Model")
print("==============================================")

# Load CSV
csv_path = "datasets/size/size_dataset.csv"
data = pd.read_csv(csv_path)

print("Dataset loaded!")
print("Total rows:", len(data))

# Keep required columns
data = data[["image_path", "head_diameter_cm"]].dropna()

# Convert diameter to numeric
data["head_diameter_cm"] = pd.to_numeric(
    data["head_diameter_cm"],
    errors="coerce"
)

data = data.dropna()

print("Valid rows:", len(data))

# Create Small / Medium / Large using dataset tertiles
q1 = data["head_diameter_cm"].quantile(0.33)
q2 = data["head_diameter_cm"].quantile(0.66)

def get_size(diameter):
    if diameter <= q1:
        return 0
    elif diameter <= q2:
        return 1
    else:
        return 2

data["size_label"] = data["head_diameter_cm"].apply(get_size)

print("\nSize boundaries:")
print("Small  <=", round(q1, 2), "cm")
print("Medium <=", round(q2, 2), "cm")
print("Large  > ", round(q2, 2), "cm")

# Image loading
IMG_SIZE = (224, 224)

images = []
labels = []

print("\nLoading images...")

for _, row in data.iterrows():

    image_path = row["image_path"]

    if not os.path.exists(image_path):
        continue

    try:
        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMG_SIZE
        )

        image = tf.keras.utils.img_to_array(image)
        image = image / 255.0

        images.append(image)
        labels.append(row["size_label"])

    except Exception as e:
        print("Skipped:", image_path)

images = np.array(images, dtype="float32")
labels = np.array(labels, dtype="int32")

print("Images loaded:", len(images))

if len(images) < 20:
    raise ValueError(
        "Not enough images loaded. Check GrowliFlower image paths."
    )

# Split
X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("\nTraining image size model...")

# MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(3, activation="softmax")(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=16
)

# Save model
os.makedirs("models", exist_ok=True)

model_path = "models/size_prediction_model.keras"

model.save(model_path)

# Save boundaries for app.py
size_info = {
    "small_max": float(q1),
    "medium_max": float(q2)
}

joblib.dump(
    size_info,
    "models/size_boundaries.pkl"
)

print("\n==============================================")
print("SIZE IMAGE MODEL TRAINED SUCCESSFULLY!")
print("==============================================")
print("Model:", model_path)
print("Boundaries: models/size_boundaries.pkl")
print("==============================================")