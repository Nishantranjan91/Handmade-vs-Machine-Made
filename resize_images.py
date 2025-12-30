import os
from PIL import Image

DATASET_DIR = "dataset"
TARGET_SIZE = (224, 224)

def resize_images(folder_path):
    print(f"📂 Processing: {folder_path}")
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, file)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(TARGET_SIZE)
                img.save(img_path)
            except Exception as e:
                print(f"❌ Error with {file}: {e}")

resize_images(os.path.join(DATASET_DIR, "Handmade"))
resize_images(os.path.join(DATASET_DIR, "Machine_made"))

print("✅ ALL images resized to 224x224 successfully")
