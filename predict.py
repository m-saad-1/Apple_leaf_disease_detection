import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# ----------------------------
# Load Class Names (Consistency)
# ----------------------------
CLASS_NAMES_PATH = "models/class_names.json"
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# ----------------------------
# Load Trained Model
# ----------------------------
MODEL_PATH = "models/leaf_model2.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ----------------------------
# Image Preprocessing Function
# ----------------------------
def preprocess_image(img_path, target_size=(224, 224), center_crop=True):
    """
    Load an image and preprocess it exactly like during training.

    Args:
        img_path (str): Path to leaf image
        target_size (tuple): Desired size for EfficientNet (default 224x224)
        center_crop (bool): Crop the center square to focus on leaf

    Returns:
        np.ndarray: Preprocessed image array ready for model prediction
    """
    # Load image with RGB mode
    img = Image.open(img_path).convert("RGB")

    # Optional center-crop to square
    if center_crop:
        min_dim = min(img.size)
        width, height = img.size
        left = (width - min_dim) / 2
        top = (height - min_dim) / 2
        right = (width + min_dim) / 2
        bottom = (height + min_dim) / 2
        img = img.crop((left, top, right, bottom))

    # Resize to target size
    img = img.resize(target_size)

    # Convert to array and expand dims for batch
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # EfficientNet preprocessing
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# ----------------------------
# Prediction Function
# ----------------------------
def predict_leaf_disease(img_path):
    """
    Predict apple leaf disease with confidence, margin, and unknown detection.

    Returns a dict:
    {
        "disease": str,
        "confidence": float,
        "margin": float,
        "note": str
    }
    """
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array, verbose=0)[0]

    # Best and second-best predictions
    sorted_idx = np.argsort(preds)[::-1]
    best_idx = sorted_idx[0]
    second_best_idx = sorted_idx[1]

    confidence = float(preds[best_idx])
    margin = float(preds[best_idx] - preds[second_best_idx])

    result = {
        "confidence": confidence,
        "margin": margin
    }

    # Out-of-distribution detection
    if confidence < 0.5 or margin < 0.1:
        result["disease"] = "Unknown / Not Apple Leaf"
        result["note"] = "The uploaded leaf may not be an Apple leaf or is unclear."
    else:
        result["disease"] = class_names[best_idx]
        if margin < 0.2 or confidence < 0.6:
            result["note"] = "Uncertain prediction – please retake image or check manually"

    return result

# ----------------------------
# Optional: Quick Test
# ----------------------------
if __name__ == "__main__":
    test_image = "test_leaf.jpg"  # Replace with a local test image path
    if os.path.exists(test_image):
        res = predict_leaf_disease(test_image)
        print("Prediction Result:")
        for k, v in res.items():
            print(f"{k}: {v}")
    else:
        print(f"Test image '{test_image}' not found.")