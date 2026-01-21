import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
import os

# Load class names (ensure same order as training)
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# Load model
MODEL_PATH = "models/leaf_model2.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess image exactly like in training"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # EfficientNet preprocessing
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def predict_leaf_disease(img_path):
    """Predict disease with confidence & margin logic"""
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array, verbose=0)[0]

    sorted_idx = np.argsort(preds)[::-1]
    best_idx = sorted_idx[0]
    second_best_idx = sorted_idx[1]

    confidence = float(preds[best_idx])
    margin = float(preds[best_idx] - preds[second_best_idx])

    result = {
        "disease": class_names[best_idx],
        "confidence": confidence,
        "margin": margin
    }

    # Low margin or low confidence detection
    if margin < 0.2 or confidence < 0.6:
        result["note"] = "Uncertain prediction – please retake image or check manually"

    return result
