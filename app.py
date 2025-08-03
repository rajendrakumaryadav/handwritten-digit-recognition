import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="✍️", layout="centered")


@st.cache_resource
def load_model(model_path: Path):
    """Loads the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        if not os.path.exists('models'):
            os.makedirs('models')
        st.info("Please make sure you have the 'mnist_model.keras' file inside a 'models' directory.")
        return None


def preprocess_image(image):
    """
    Preprocesses the uploaded image to the format required by the MNIST model.
    - Converts to grayscale
    - Inverts colors (MNIST has white digits on a black background)
    - Resizes to 28x28 pixels
    - Normalizes pixel values
    - Reshapes for model input
    """
    grayscale_image = ImageOps.grayscale(image)
    
    inverted_image = ImageOps.invert(grayscale_image)
    resized_image = inverted_image.resize((28, 28))
    img_array = np.array(resized_image)
    normalized_img = img_array / 255.0
    reshaped_img = normalized_img.reshape(1, 28, 28)
    
    return reshaped_img


st.title("✍️ MNIST Digit Recognizer")
st.write("Upload an image of a handwritten digit (0-9), and the model will predict what it is. "
         "For best results, use a clear image with the digit centered.")

MODEL_DIR = Path('models')
MODEL_FILENAME = 'mnist_model.keras'
model_path_ = Path.joinpath(MODEL_DIR, MODEL_FILENAME)

model_ = load_model(model_path_)

if model_:
    uploaded_file = st.file_uploader("Choose a digit image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.write("### Prediction")
            with st.spinner("Classifying..."):
                processed_image = preprocess_image(image)
                prediction = model_.predict(processed_image)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)
                st.success(f"I think this is the digit: **{predicted_digit}**")
                st.info(f"Confidence: **{confidence:.2%}**")
                st.write("### Prediction Probabilities")
                st.bar_chart(prediction.flatten())
