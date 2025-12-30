import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("handmade_vs_machine_model.h5")

# Image size (same as training)
IMG_SIZE = (224, 224)

# Class names (same order as training)
class_names = ["Handmade", "Machine_made"]

# Images list
images = [
    "test1.jpg",
    "test2.jpg",
    "test3.jpg"
]

print("\n🔍 Running predictions on multiple images:\n")

for img_name in images:
    if not os.path.exists(img_name):
        print(f"❌ {img_name} not found")
        continue

    img = image.load_img(img_name, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]

    print(f"🖼️ {img_name} → ✅ {predicted_class}")
