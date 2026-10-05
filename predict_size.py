import os
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
    "size_prediction_model.keras"
)

IMG_SIZE = (224, 224)

SIZE_CLASSES = [
    "Small",
    "Medium",
    "Large"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading size model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Size model loaded successfully.")


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_size(image_path):

    if not os.path.isfile(image_path):
        print("\nERROR: Image file not found.")
        print(image_path)
        return

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image = np.array(image, dtype=np.float32)
    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0]

    index = int(np.argmax(prediction))

    predicted_size = SIZE_CLASSES[index]
    confidence = float(prediction[index]) * 100

    print("\n" + "=" * 50)
    print("SIZE PREDICTION RESULT")
    print("=" * 50)

    print(f"Predicted Size : {predicted_size}")
    print(f"Confidence     : {confidence:.2f}%")

    print("=" * 50)


# ============================================================
# MAIN
# ============================================================

print("\n" + "=" * 50)
print("CAULIFLOWER SIZE PREDICTION")
print("=" * 50)

image_path = input(
    "\nEnter cauliflower image path: "
).strip()

# Remove quotes from Windows Copy as Path
image_path = image_path.strip('"')

predict_size(image_path)