"""
Fallback Grad-CAM Implementation (for backward compatibility)

This module is deprecated. Use gradcam_disease_focused.py instead for better results.
"""

import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from typing import Tuple, Optional, Dict, Any
from datetime import datetime


def generate_enhanced_gradcam(
    image_path: str,
    model: tf.keras.Model,
    visualization_mode: str = "blend",
    output_dir: str = "static/gradcam"
) -> Dict[str, Any]:
    """
    Fallback wrapper using disease-focused Grad-CAM.
    
    Args:
        image_path: Path to input image
        model: Disease classification model
        visualization_mode: 'blend', 'contour', or 'highlight'
        output_dir: Output directory for visualization
        
    Returns:
        Dictionary with success status and gradcam_image path
    """
    try:
        # Import the new disease-focused implementation
        from .gradcam_disease_focused import generate_disease_focused_gradcam
        
        gradcam_path = generate_disease_focused_gradcam(
            image_path=image_path,
            model=model,
            output_dir=output_dir,
            mode=visualization_mode
        )
        
        if gradcam_path:
            return {
                "success": True,
                "gradcam_image": gradcam_path,
                "error": None
            }
        else:
            return {
                "success": False,
                "gradcam_image": None,
                "error": "Failed to generate Grad-CAM"
            }
    
    except Exception as e:
        return {
            "success": False,
            "gradcam_image": None,
            "error": f"Grad-CAM generation error: {str(e)}"
        }


