"""
Shared model loading utility for all evaluation scripts.
Handles the Keras 2/3 compatibility issue: the saved .keras files
use keras.src.models.functional (Keras 3 format), so we need
custom_objects or safe_mode=False to load them under TF 2.14.
"""
import os
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def load_keras_model(model_path):
    """
    Load a .keras model robustly, handling Keras 2/3 format differences.
    Tries multiple loading strategies to maximize compatibility.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Strategy 1: safe_mode=False (allows arbitrary Keras class loading)
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info(f"Loaded (strategy 1): {os.path.basename(model_path)}")
        return model
    except Exception as e1:
        logger.warning(f"Strategy 1 failed: {e1}")

    # Strategy 2: custom_objects with Functional model registered
    try:
        from tensorflow.python.keras.saving import hdf5_format
        model = tf.keras.models.load_model(model_path, compile=False,
                                            custom_objects={'Functional': tf.keras.Model})
        logger.info(f"Loaded (strategy 2): {os.path.basename(model_path)}")
        return model
    except Exception as e2:
        logger.warning(f"Strategy 2 failed: {e2}")

    # Strategy 3: Load weights only if model architecture is known
    raise RuntimeError(
        f"Cannot load model: {model_path}\n"
        f"This model was saved with Keras 3 but the environment has Keras 2.\n"
        f"To fix: pip install keras>=3.0 OR use the same TF version that trained the model."
    )
