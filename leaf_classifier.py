"""
Stage 2 Classifier: Apple Leaf Disease Classification
Classifies apple leaves into specific disease categories:
- Apple___Apple_scab
- Apple___Black_rot
- Apple___Cedar_apple_rust
- Apple___healthy
"""

import tensorflow as tf
import numpy as np
import json
import os
from utils.image_processing import preprocess_image
from config import (
    STAGE2_MODEL_PATH,
    STAGE2_CLASS_NAMES_PATH,
    get_stage2_config,
    get_disease_info,
    get_confidence_level
)

# Load Stage 2 Model and Class Names (robust initialization)
stage2_model = None
stage2_class_names = []
stage2_load_error = None

try:
    if not os.path.exists(STAGE2_MODEL_PATH):
        raise FileNotFoundError(f"Stage 2 model not found at {STAGE2_MODEL_PATH}")

    if not os.path.exists(STAGE2_CLASS_NAMES_PATH):
        raise FileNotFoundError(f"Stage 2 class names not found at {STAGE2_CLASS_NAMES_PATH}")

    stage2_model = tf.keras.models.load_model(STAGE2_MODEL_PATH)

    with open(STAGE2_CLASS_NAMES_PATH, "r") as f:
        stage2_class_names = json.load(f)

except Exception as e:
    stage2_load_error = str(e)

# Load configuration
STAGE2_CONF = get_stage2_config()


def classify_stage2(img_path, confidence_threshold=None, margin_threshold=None):
    """
    Stage 2 Classification: Identify specific disease or confirm healthy status.

    Args:
        img_path (str): Path to the image file
        confidence_threshold (float): Minimum confidence for a valid prediction (uses config if None)
        margin_threshold (float): Minimum margin between top 2 predictions (uses config if None)

    Returns:
        dict: {
            "stage": 2,
            "disease": str,  # Disease class name
            "disease_display": str,  # Human-readable disease name
            "confidence": float,
            "margin": float,
            "description": str,
            "message": str
        }
    """
    try:
        if stage2_model is None:
            return {
                "stage": 2,
                "disease": "Error",
                "disease_display": "Error",
                "confidence": 0.0,
                "margin": 0.0,
                "description": "",
                "message": f"Stage 2 model unavailable: {stage2_load_error}"
            }

        # Use config thresholds if not provided
        if confidence_threshold is None:
            confidence_threshold = STAGE2_CONF["confidence_threshold"]
        if margin_threshold is None:
            margin_threshold = STAGE2_CONF["margin_threshold"]

        # Preprocess image with target size from config
        img_array = preprocess_image(
            img_path,
            target_size=STAGE2_CONF["target_size"],
            center_crop=STAGE2_CONF["center_crop"]
        )

        # Get predictions
        predictions = stage2_model.predict(img_array, verbose=0)[0]

        # Find top 2 predictions
        sorted_indices = np.argsort(predictions)[::-1]
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1]

        confidence = float(predictions[best_idx])
        margin = float(predictions[best_idx] - predictions[second_best_idx])

        # Get predicted disease
        predicted_disease = stage2_class_names[best_idx]
        
        # Get disease information from config
        disease_info = get_disease_info(predicted_disease)
        display_name = disease_info["display_name"]
        description = disease_info["description"]

        # Determine if prediction is reliable
        is_confident = confidence >= confidence_threshold and margin >= margin_threshold

        result = {
            "stage": 2,
            "disease": predicted_disease,
            "disease_display": display_name,
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "description": description,
            "message": ""
        }

        # Evaluate the prediction
        if not is_confident:
            result["message"] = (
                f"Prediction made with moderate confidence (confidence: {confidence:.2%}, margin: {margin:.4f}). "
                "Results should be verified by a domain expert."
            )
        elif predicted_disease == "Apple___healthy":
            result["message"] = f"The leaf is classified as {display_name}. Stage 1 indicated a diseased leaf, but Stage 2 confirms it is healthy."
        else:
            result["message"] = f"Disease identified: {display_name}. {description}"

        return result

    except Exception as e:
        return {
            "stage": 2,
            "disease": "Error",
            "disease_display": "Error",
            "confidence": 0.0,
            "margin": 0.0,
            "description": "",
            "message": f"Error during Stage 2 classification: {str(e)}"
        }
