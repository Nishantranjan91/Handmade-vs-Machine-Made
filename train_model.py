import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Load base model
base_model=MobileNetV2(
weights="imagenet",
include_top=False,
input_shape=(224,224,3)
)
# Freeze base model
base_model.trainable=False
# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

#Compile
model.compile(
    optimizer=Adam(learning_rate=0.0001),
loss="binary_crossentropy",
metrics=["accuracy"]
)

#Data generators (No augmentation for now)
train_datagen=ImageDataGenerator(rescale=1./255)
val_datagen=ImageDataGenerator(rescale=1./255)
train_data=train_datagen.flow_from_directory(
"dataset_split/train",
target_size=(224,224),
batch_size=32,
class_mode="binary"
)
val_data=val_datagen.flow_from_directory(
"dataset_split/val",
target_size=(224,224),
batch_size=32,
class_mode="binary"
)
#Train
history=model.fit(
train_data,
validation_data=val_data,
epochs=5
)

#Save model
model.save("mobilenetv2_handmade_vs_machine.h5")
print("Training complete & model saved")
import matplotlib.pyplot as plt

# Accuracy graph
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("training_accuracy.png")
plt.close()


# Loss graph
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("training_loss.png")
plt.close()

