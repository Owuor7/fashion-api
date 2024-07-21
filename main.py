import os
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Ensure the model file exists
model_path = 'fash_cnn_model.keras'
if not os.path.exists(model_path):
    raise ValueError(f"Model file not found: {model_path}")

# Load the trained model
model = load_model(model_path)

# Define class names for CIFAR-10 dataset
class_names = [
  "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def preprocess_image(image_bytes):

    # Open the image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize the image to 32x32
    img = img.resize((28, 28))

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Normalize the image
    img_array = img_array.astype('float32') / 255.0

    # Expand dimensions to match the model input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
async def read_root():

    return {
        "message": "Welcome to the MNIST fashion classification API!",
        "instructions": {
            "POST /predict/": "Upload a RGB image of any fashion design to get the predicted class."
        }
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    image_bytes = await file.read()
    # Preprocess the image
    img_array = preprocess_image(image_bytes)
    # Make predictions
    predictions = model.predict(img_array)
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = class_names[predicted_class[0]]
    return {"predicted_label": predicted_label}

