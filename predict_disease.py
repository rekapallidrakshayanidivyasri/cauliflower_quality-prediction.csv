import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "disease_model.keras"
)

CLASS_FILE = os.path.join(
    BASE_DIR,
    "models",
    "disease_classes.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading disease model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Disease model loaded successfully.")


# ============================================================
# LOAD EXACT TRAINING CLASSES
# ============================================================

if os.path.exists(CLASS_FILE):

    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        class_data = json.load(f)

    if isinstance(class_data, dict):
        CLASS_NAMES = [
            class_data[str(i)]
            for i in range(len(class_data))
        ]
    else:
        CLASS_NAMES = class_data

else:

    CLASS_NAMES = [
        "Alternaria Brassicae",
        "Bacterial Soft Rot",
        "Bacterial Spot",
        "Black Rot",
        "Healthy",
        "Purple Tinges"
    ]


print("\nDisease Classes:")

for i, name in enumerate(CLASS_NAMES):
    print(i, "->", name)


# ============================================================
# IMAGE PREDICTION FUNCTION
# ============================================================

def predict_disease(image_path):

    if not os.path.exists(image_path):

        print("\nERROR: Image file not found.")
        return

    print("\nLoading image...")

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # IMPORTANT:
    # Do NOT divide by 255 here.
    # The trained model already contains Rescaling.

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_disease = CLASS_NAMES[predicted_index]

    confidence = float(
        predictions[predicted_index] * 100
    )


    # ========================================================
    # RESULT
    # ========================================================

    print("\n" + "=" * 50)
    print("DISEASE PREDICTION RESULT")
    print("=" * 50)

    print(
        "Predicted Disease:",
        predicted_disease
    )

    print(
        "Confidence:",
        f"{confidence:.2f}%"
    )

    print("=" * 50)


# ============================================================
# MAIN
# ============================================================

print("\n" + "=" * 50)
print("CAULIFLOWER DISEASE PREDICTION")
print("=" * 50)

image_path = input(
    "\nEnter cauliflower image path: "
).strip()

# Remove quotes if Windows Copy as Path added them
image_path = image_path.strip('"')

predict_disease(image_path)