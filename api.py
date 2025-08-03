import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = 'models'
MODEL_FILENAME = 'mnist_model.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

ml_models = {}


@asynccontextmanager
async def lifespan(app):
    """
    Asynchronous context manager to handle application startup and shutdown events.
    This is the recommended way to load models in modern FastAPI.
    """
    # Startup: Load the model
    print("--- Loading model ---")
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. "
                                    "Please ensure 'mnist_model.keras' is in the 'models' directory.")
        ml_models['mnist_predictor'] = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        ml_models['mnist_predictor'] = None
    
    yield
    
    ml_models.clear()
    print("🧹 Model resources cleared.")


app = FastAPI(title="MNIST Digit Recognizer API",
              description="An API to predict handwritten digits from images using a trained Keras model.",
              version="1.0.0", lifespan=lifespan)


class PredictionResponse(BaseModel):
    """Defines the structure of the API response."""
    predicted_digit: int = Field(example=7, description="The digit predicted by the model.")
    confidence: float = Field(example=0.998, description="The probability score of the prediction.")
    probabilities: list[float] = Field(example=[0.001, 0.0, 0.002, 0.01, 0.0, 0.0, 0.0, 0.98, 0.0, 0.007],
                                       description="List of probabilities for each digit (0-9).")
    prediction_time_ms: float = Field(example=25.5, description="Time taken for the prediction in milliseconds.")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocesses the image bytes to the format required by the MNIST model.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        grayscale_image = ImageOps.grayscale(image)
        
        inverted_image = ImageOps.invert(grayscale_image)
        
        resized_image = inverted_image.resize((28, 28))
        
        img_array = np.array(resized_image) / 255.0
        
        return img_array.reshape(1, 28, 28)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file. Error: {e}")


@app.post("/predict/", response_model=PredictionResponse)
async def predict_digit(file: UploadFile = File(..., description="An image file of a handwritten digit.")):
    """
    Accepts an image file, preprocesses it, and returns the predicted digit.
    """
    if ml_models.get('mnist_predictor') is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please check server logs.")
    
    image_bytes = await file.read()
    
    start_time = time.time()
    
    processed_image = preprocess_image(image_bytes)
    
    prediction = ml_models['mnist_predictor'].predict(processed_image)
    
    end_time = time.time()
    
    probabilities = prediction.flatten()
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    
    prediction_time_ms = (end_time - start_time) * 1000
    
    return PredictionResponse(predicted_digit=predicted_digit, confidence=confidence,
                              probabilities=probabilities.tolist(), prediction_time_ms=prediction_time_ms)


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the MNIST Recognizer API! Go to /docs for the API documentation."}
