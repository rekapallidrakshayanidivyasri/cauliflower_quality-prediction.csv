import os
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Project folder:")
print(BASE_DIR)


# ============================================================
# 2. QUALITY DATASET PATH
# ============================================================

QUALITY_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "quality"
)

print("\nQuality dataset path:")
print(QUALITY_DIR)


# ============================================================
# 3. CHECK DATASET
# ============================================================

if not os.path.exists(QUALITY_DIR):
    raise FileNotFoundError(
        "\nQuality dataset was not found.\n"
        "Please check the quality folder path."
    )

print("\nQuality dataset found successfully!")


# ============================================================
# 4. IMAGE SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# 5. LOAD TRAINING DATA
# ============================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    QUALITY_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ============================================================
# 6. LOAD VALIDATION DATA
# ============================================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    QUALITY_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ============================================================
# 7. GET QUALITY CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nQuality Classes:")

for number, name in enumerate(class_names):
    print(number, "->", name)


# ============================================================
# 8. IMPROVE DATA LOADING
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 9. LOAD MOBILENETV2
# ============================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

print("\nMobileNetV2 loaded successfully!")


# ============================================================
# 10. CREATE QUALITY MODEL
# ============================================================

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Rescaling(
        1.0 / 127.5,
        offset=-1
    ),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


# ============================================================
# 11. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nQuality model compiled successfully!")


# ============================================================
# 12. CREATE MODELS FOLDER
# ============================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 13. MODEL SAVE PATH
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "quality_model.keras"
)

print("\nModel will be saved at:")
print(MODEL_PATH)


# ============================================================
# 14. MODEL CHECKPOINT
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


# ============================================================
# 15. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ============================================================
# 16. TRAIN MODEL
# ============================================================

print("\n========================================")
print("Starting Quality Model Training")
print("========================================\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# ============================================================
# 17. EVALUATE MODEL
# ============================================================

loss, accuracy = model.evaluate(
    validation_dataset
)


# ============================================================
# 18. FINAL RESULT
# ============================================================

print("\n========================================")
print("Quality Model Training Complete")
print("========================================")

print(
    "Validation Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nModel saved at:")
print(MODEL_PATH)