import os
import json
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("\n" + "=" * 60)
print("SMART CAULIFLOWER - DISEASE MODEL TRAINING")
print("=" * 60)

print("\nProject folder:")
print(BASE_DIR)


# ============================================================
# 2. FIND DISEASE DATASET AUTOMATICALLY
# ============================================================

DISEASE_ROOT = os.path.join(
    BASE_DIR,
    "datasets",
    "diseases"
)

if not os.path.exists(DISEASE_ROOT):
    raise FileNotFoundError(
        f"\nDisease root folder not found:\n{DISEASE_ROOT}"
    )

print("\nDisease root found:")
print(DISEASE_ROOT)


# Search recursively for the folder containing the 6 class folders
EXPECTED_CLASS_COUNT = 6

DATASET_DIR = None

for root, dirs, files in os.walk(DISEASE_ROOT):

    valid_subfolders = []

    for folder in dirs:
        folder_path = os.path.join(root, folder)

        try:
            if any(
                os.path.isfile(
                    os.path.join(folder_path, f)
                )
                for f in os.listdir(folder_path)
            ):
                valid_subfolders.append(folder)
        except:
            pass

    if len(valid_subfolders) == EXPECTED_CLASS_COUNT:
        DATASET_DIR = root
        break


if DATASET_DIR is None:
    raise FileNotFoundError(
        "\nCould not automatically find the disease category folder."
    )


print("\n" + "=" * 60)
print("ACTUAL DISEASE DATASET")
print("=" * 60)

print(DATASET_DIR)


# ============================================================
# 3. GET ACTUAL CLASS NAMES FROM DATASET
# ============================================================

class_names = sorted([
    folder
    for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(
        os.path.join(DATASET_DIR, folder)
    )
])


print("\n" + "=" * 60)
print("ACTUAL DISEASE CLASSES")
print("=" * 60)

for i, name in enumerate(class_names):
    print(i, "->", name)


# ============================================================
# 4. CHECK CLASS COUNT
# ============================================================

if len(class_names) != 6:

    raise ValueError(
        f"\nExpected 6 disease classes, "
        f"but found {len(class_names)}:\n"
        + "\n".join(class_names)
    )


print("\nAll 6 disease classes detected successfully!")


# ============================================================
# 5. IMAGE SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# 6. LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# 7. LOAD VALIDATION DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING VALIDATION DATA")
print("=" * 60)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 8. VERIFY TENSORFLOW CLASS ORDER
# ============================================================

tensorflow_classes = train_dataset.class_names

print("\n" + "=" * 60)
print("TENSORFLOW CLASS ORDER")
print("=" * 60)

for i, name in enumerate(tensorflow_classes):
    print(i, "->", name)


if tensorflow_classes != class_names:

    raise ValueError(
        "\nDataset class order mismatch.\n"
        f"Folders: {class_names}\n"
        f"TensorFlow: {tensorflow_classes}"
    )


print("\nClass order verified successfully!")


# ============================================================
# 9. COUNT IMAGES
# ============================================================

print("\n" + "=" * 60)
print("IMAGE COUNTS")
print("=" * 60)

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

total_images = 0

for class_name in class_names:

    class_folder = os.path.join(
        DATASET_DIR,
        class_name
    )

    count = 0

    for filename in os.listdir(class_folder):

        if filename.lower().endswith(image_extensions):
            count += 1

    total_images += count

    print(
        f"{class_name}: {count} images"
    )


print("\nTotal images:", total_images)


# ============================================================
# 10. PREFETCH
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)


# ============================================================
# 11. DATA AUGMENTATION
# ============================================================

data_augmentation = models.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.10
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        )
    ],
    name="data_augmentation"
)


# ============================================================
# 12. LOAD MOBILENETV2
# ============================================================

print("\n" + "=" * 60)
print("LOADING MOBILENETV2")
print("=" * 60)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

print("MobileNetV2 loaded successfully!")


# ============================================================
# 13. CREATE DISEASE MODEL
# ============================================================

model = models.Sequential(
    [
        layers.Input(
            shape=(224, 224, 3)
        ),

        data_augmentation,

        layers.Rescaling(
            1.0 / 127.5,
            offset=-1
        ),

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dropout(0.30),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.20),

        layers.Dense(
            len(class_names),
            activation="softmax"
        )
    ],
    name="cauliflower_disease_model"
)


# ============================================================
# 14. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nDisease model compiled successfully!")


# ============================================================
# 15. MODEL DIRECTORY
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
# 16. MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "disease_model.keras"
)

CLASS_FILE = os.path.join(
    MODEL_DIR,
    "disease_classes.json"
)


print("\nModel will be saved at:")
print(MODEL_PATH)


# ============================================================
# 17. SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_names,
        f,
        indent=4,
        ensure_ascii=False
    )


print("\nDisease class file saved at:")
print(CLASS_FILE)


# ============================================================
# 18. CHECKPOINT
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


# ============================================================
# 19. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 20. START TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING DISEASE MODEL TRAINING")
print("=" * 60)

print("Epochs: 30")
print("Batch size:", BATCH_SIZE)
print("Image size:", IMG_SIZE)

print("\nTraining started...\n")


history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=30,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# ============================================================
# 21. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING DISEASE MODEL")
print("=" * 60)

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)


# ============================================================
# 22. SAVE FINAL MODEL
# ============================================================

model.save(MODEL_PATH)


# ============================================================
# 23. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("DISEASE MODEL TRAINING COMPLETE")
print("=" * 60)

print(
    "Validation Loss:",
    round(loss, 4)
)

print(
    "Validation Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


print("\nModel saved at:")
print(MODEL_PATH)


# ============================================================
# 24. FINAL CLASS MAPPING
# ============================================================

print("\n" + "=" * 60)
print("FINAL DISEASE CLASS MAPPING")
print("=" * 60)

for i, name in enumerate(class_names):

    print(
        i,
        "->",
        name
    )


# ============================================================
# 25. COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("DISEASE TRAINING FINISHED")
print("=" * 60)

print("Next: Test the disease model with a real image.")
print("=" * 60)