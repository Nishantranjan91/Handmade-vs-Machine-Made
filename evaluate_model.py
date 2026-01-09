import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
model = tf.keras.models.load_model("mobilenetv2_handmade_vs_machine.h5")

# Validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)

val_data = val_datagen.flow_from_directory(
    "dataset_split/val",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Predictions
y_true = val_data.classes
y_pred_prob = model.predict(val_data)
y_pred = (y_pred_prob > 0.5).astype(int).ravel()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Handmade", "Machine-Made"]))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Confusion matrix values (already calculated)
cm = confusion_matrix(y_true, y_pred)

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Handmade", "Machine-Made"],
    yticklabels=["Handmade", "Machine-Made"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Handmade vs Machine-Made")

# Save image
plt.savefig("confusion_matrix.png")
plt.show()
