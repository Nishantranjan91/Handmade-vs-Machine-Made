import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("mobilenetv2_handmade_vs_machine.h5")

IMG_SIZE = 224

def predict_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return "🟢 Machine-Made"
    else:
        return "🟠 Handmade"

# Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(label="Prediction"),
    title="Handmade vs Machine-Made Image Classifier",
    description="Upload an artwork image to check whether it is Handmade or Machine-Made"
)

interface.launch(share=True)
