import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("mobilenetv2_handmade_vs_machine.h5")

# Folder containing test images
test_folder = "test_images"

IMG_SIZE = 224

print("🔍 Predicting images from folder:", test_folder)
print("-" * 40)

for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)

    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        label = "Machine-made" if prediction > 0.5 else "Handmade"

        print(f"{img_name:25} → {label}")
