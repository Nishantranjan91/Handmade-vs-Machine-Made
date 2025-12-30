import os
import shutil
import random

# Paths
BASE_DIR = "dataset"
CLASSES = ["Handmade", "Machine_made"]
OUTPUT_DIR = "dataset_split"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_dirs():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

def split_images():
    for cls in CLASSES:
        src_dir = os.path.join(BASE_DIR, cls)
        images = os.listdir(src_dir)
        random.shuffle(images)

        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, files in splits.items():
            for file in files:
                src_path = os.path.join(src_dir, file)
                dst_path = os.path.join(OUTPUT_DIR, split, cls, file)
                shutil.copy(src_path, dst_path)

        print(f"✅ {cls}: {len(images)} images split")

if __name__ == "__main__":
    create_dirs()
    split_images()
    print("🎉 Dataset successfully split into train / val / test")

