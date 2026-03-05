"""
Stage 1 Classifier: Apple Leaf Detection
Classifies images into three categories:
- Apple_Healthy (healthy apple leaf)
- Apple_Diseased (diseased apple leaf)
- Not_Apple_Leaf (not an apple leaf)
"""

import tensorflow as tf
import numpy as np
import json
import os
from utils.image_processing import preprocess_image
from config import (
    STAGE1_MODEL_PATH,
    STAGE1_CLASS_NAMES_PATH,
    get_stage1_config,
    get_confidence_level
)

# Load Stage 1 Model and Class Names (robust initialization)
stage1_model = None
stage1_class_names = []
stage1_load_error = None

try:
    if not os.path.exists(STAGE1_MODEL_PATH):
        raise FileNotFoundError(f"Stage 1 model not found at {STAGE1_MODEL_PATH}")

    if not os.path.exists(STAGE1_CLASS_NAMES_PATH):
        raise FileNotFoundError(f"Stage 1 class names not found at {STAGE1_CLASS_NAMES_PATH}")

    stage1_model = tf.keras.models.load_model(STAGE1_MODEL_PATH)

    with open(STAGE1_CLASS_NAMES_PATH, "r") as f:
        stage1_class_names = json.load(f)

except Exception as e:
    stage1_load_error = str(e)

# Load configuration
STAGE1_CONF = get_stage1_config()

# Stage 1 class mapping
STAGE1_CLASSES = {
    "Apple_Diseased": 0,
    "Apple_Healthy": 1,
    "Not_Apple_Leaf": 2
}

STAGE1_CLASSES_REVERSE = {v: k for k, v in STAGE1_CLASSES.items()}


def classify_stage1(img_path, confidence_threshold=None, margin_threshold=None):
    """
    Stage 1 Classification: Determine if image is an apple leaf and its health status.

    Args:
        img_path (str): Path to the image file
        confidence_threshold (float): Minimum confidence for a valid prediction (uses config if None)
        margin_threshold (float): Minimum margin between top 2 predictions (uses config if None)

    Returns:
        dict: {
            "stage": 1,
            "category": str,  # "Apple_Healthy", "Apple_Diseased", or "Not_Apple_Leaf"
            "confidence": float,
            "margin": float,
            "is_apple_leaf": bool,
            "needs_stage2": bool,
            "message": str
        }
    """
    try:
        if stage1_model is None:
            return {
                "stage": 1,
                "category": "Error",
                "confidence": 0.0,
                "margin": 0.0,
                "is_apple_leaf": False,
                "needs_stage2": False,
                "message": f"Stage 1 model unavailable: {stage1_load_error}"
            }

        # Use config thresholds if not provided
        if confidence_threshold is None:
            confidence_threshold = STAGE1_CONF["confidence_threshold"]
        if margin_threshold is None:
            margin_threshold = STAGE1_CONF["margin_threshold"]

        # Preprocess image with target size from config
        img_array = preprocess_image(
            img_path,
            target_size=STAGE1_CONF["target_size"],
            center_crop=STAGE1_CONF["center_crop"]
        )

        # Get predictions
        predictions = stage1_model.predict(img_array, verbose=0)[0]

        # Find top 2 predictions
        sorted_indices = np.argsort(predictions)[::-1]
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1]

        confidence = float(predictions[best_idx])
        margin = float(predictions[best_idx] - predictions[second_best_idx])

        # Get predicted class
        predicted_class = stage1_class_names[best_idx]

        # Determine if prediction is reliable
        is_confident = confidence >= confidence_threshold and margin >= margin_threshold

        result = {
            "stage": 1,
            "category": predicted_class,
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "is_apple_leaf": predicted_class != "Not_Apple_Leaf",
            "needs_stage2": False,
            "message": ""
        }

        # Evaluate the prediction
        if not is_confident:
            result["message"] = (
                f"Low confidence detection (confidence: {confidence:.2%}, margin: {margin:.4f}). "
                "Please upload a clearer image of an apple leaf."
            )
        elif predicted_class == "Not_Apple_Leaf":
            result["message"] = "This image does not appear to be an apple leaf. Please upload an apple leaf image."
        elif predicted_class == "Apple_Healthy":
            result["message"] = "This apple leaf appears to be healthy."
        elif predicted_class == "Apple_Diseased":
            result["needs_stage2"] = True
            result["message"] = "A disease has been detected. Analyzing specific disease type..."

        return result

    except Exception as e:
        return {
            "stage": 1,
            "category": "Error",
            "confidence": 0.0,
            "margin": 0.0,
            "is_apple_leaf": False,
            "needs_stage2": False,
            "message": f"Error during Stage 1 classification: {str(e)}"
        }
