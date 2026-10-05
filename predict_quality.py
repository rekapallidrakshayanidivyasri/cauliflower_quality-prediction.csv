import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Project path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Quality dataset path
QUALITY_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "quality"
)

# Model path
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "quality_model.keras"
)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Automatically get image size from trained model
IMG_SIZE = (
    model.input_shape[1],
    model.input_shape[2]
)

print("====================================")
print("CAULIFLOWER QUALITY PREDICTION")
print("====================================")
print("Model Input Size:", IMG_SIZE)

# Get quality class names
class_names = []

for folder in os.listdir(QUALITY_DIR):
    folder_path = os.path.join(QUALITY_DIR, folder)

    if os.path.isdir(folder_path):
        class_names.append(folder)

class_names = sorted(class_names)

print("\nQuality Classes:")

for i, name in enumerate(class_names):
    print(i, ":", name)

# Ask for image path
image_path = input("\nEnter cauliflower image path: ").strip().strip('"')

# Check image exists
if not os.path.exists(image_path):
    print("\nERROR: Image path not found.")
    print("Please check the image path and try again.")
    exit()

# Load image
image = load_img(
    image_path,
    target_size=IMG_SIZE
)

# Convert image to array
image_array = img_to_array(image)

# Normalize
image_array = image_array / 255.0

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# Prediction
prediction = model.predict(image_array, verbose=0)

predicted_index = int(np.argmax(prediction[0]))
confidence = float(prediction[0][predicted_index]) * 100

print("\n====================================")
print("QUALITY PREDICTION RESULT")
print("====================================")

if predicted_index < len(class_names):
    predicted_quality = class_names[predicted_index]

    print("Predicted Quality :", predicted_quality)
    print("Confidence        :", round(confidence, 2), "%")
else:
    print("Prediction index does not match class names.")

print("====================================")