"""
Image Preprocessing Utilities for Apple Leaf Disease Detection
Provides consistent preprocessing for both Stage 1 and Stage 2 models.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image


def preprocess_image(img_path, target_size=(224, 224), center_crop=True, normalize=True):
    """
    Load and preprocess an image for model inference.
    
    This function ensures consistent preprocessing for both Stage 1 and Stage 2 models.
    Uses EfficientNet preprocessing by default (which requires no manual normalization).

    Args:
        img_path (str): Path to the image file
        target_size (tuple): Target image dimensions (default 224x224 for EfficientNet)
        center_crop (bool): Whether to center-crop to a square (default True)
        normalize (bool): Whether to apply EfficientNet preprocessing (default True)

    Returns:
        np.ndarray: Preprocessed image array ready for model inference (shape: (1, H, W, 3))
    """
    # Load image in RGB mode
    img = Image.open(img_path).convert("RGB")

    # Center-crop to square (focuses on leaf, removes background)
    if center_crop:
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        img = img.crop((left, top, right, bottom))

    # Resize to model's expected input size
    img = img.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Apply EfficientNet preprocessing (scales to [-1, 1] range)
    if normalize:
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    return img_array


def preprocess_for_model(img_path, target_size=(224, 224)):
    """
    Legacy function for backward compatibility.
    Use preprocess_image() for new code.
    """
    return preprocess_image(img_path, target_size=target_size, center_crop=True, normalize=True)
