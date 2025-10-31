import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 🔹 Path to best model
MODEL_PATH = r"C:\Guvi\Fish Classification\Tranied models\VGG16\VGG16_final.h5"

# 🔹 Load the model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# 🔹 Class names (from your dataset)
CLASS_NAMES = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload a fish image and the model will predict its species.")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("⏳ Classifying...")
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"✅ Predicted: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}%")
