from fiftyone.utils.huggingface import load_from_hub
import csv
import os
import re

print("==============================================")
print("Smart Cauliflower - Size Dataset")
print("==============================================")

print("\nLoading GrowliFlower dataset...")

dataset = load_from_hub("Voxel51/GrowliFlower")

print("Dataset loaded successfully!")
print("Total samples:", len(dataset))

print("\nLoading reference_growth view...")

reference_view = dataset.load_saved_view("reference_growth")

print("Reference samples:", len(reference_view))

output_file = "datasets/size/size_dataset.csv"

os.makedirs("datasets/size", exist_ok=True)

rows = []

print("\nCollecting head diameter measurements...")

for sample in reference_view:

    value = sample.in_situ_head_diameter

    if value is None:
        continue

    value = str(value).strip()

    if value == "":
        continue

    numbers = re.findall(r"\d+(?:\.\d+)?", value)

    if not numbers:
        continue

    numbers = [float(x) for x in numbers]

    diameter = sum(numbers) / len(numbers)

    rows.append({
        "image_path": sample.filepath,
        "head_diameter_cm": diameter,
        "field": sample.field,
        "day_after_planting": sample.day_after_planting
    })

print("\nValid measurements collected:", len(rows))

with open(output_file, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image_path",
            "head_diameter_cm",
            "field",
            "day_after_planting"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print("\n==============================================")
print("SIZE DATASET CREATED SUCCESSFULLY!")
print("==============================================")
print("CSV file:", output_file)
print("Total valid rows:", len(rows))
print("==============================================")