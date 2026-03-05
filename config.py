"""
Configuration Module for Apple Leaf Disease Detection System
All system parameters and thresholds are centralized here for easy customization.
"""

import os

# ====================
# FLASK CONFIGURATION
# ====================

# File upload settings
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE_MB = 16  # Maximum uploaded file size in MB

# Flask server settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5001
FLASK_DEBUG = True
SECRET_KEY = 'apple-leaf-detection-secret-2026'

# ====================
# MODEL PATHS
# ====================

# Stage 1: Apple Leaf Detection
STAGE1_MODEL_PATH = "models/stage1_model.keras"
STAGE1_CLASS_NAMES_PATH = "models/stage1_class_names.json"

# Stage 2: Disease Classification
STAGE2_MODEL_PATH = "models/leaf_model2.keras"
STAGE2_CLASS_NAMES_PATH = "models/class_names.json"

# ====================
# CLASSIFICATION THRESHOLDS
# ====================

# Stage 1: Apple Leaf Detection Thresholds
# Adjust these to make Stage 1 more or less strict
STAGE1_CONFIG = {
    "confidence_threshold": 0.70,    # Minimum confidence (0.0-1.0)
    "margin_threshold": 0.15,        # Minimum margin between top 2 predictions (0.0-1.0)
    "target_size": (224, 224),       # Image size for EfficientNet
    "center_crop": True,              # Whether to center-crop before resizing
}

# Stage 2: Disease Classification Thresholds
# Higher thresholds for more precise disease identification
STAGE2_CONFIG = {
    "confidence_threshold": 0.75,    # Minimum confidence (0.0-1.0)
    "margin_threshold": 0.20,        # Minimum margin between top 2 predictions (0.0-1.0)
    "target_size": (224, 224),       # Image size for EfficientNet
    "center_crop": True,              # Whether to center-crop before resizing
}

# ====================
# DISEASE INFORMATION
# ====================

# Disease display names (human-readable versions)
DISEASE_DISPLAY_NAMES = {
    "Apple___Apple_scab": "Apple Scab",
    "Apple___Black_rot": "Black Rot",
    "Apple___Cedar_apple_rust": "Cedar Apple Rust",
    "Apple___healthy": "Healthy"
}

# Disease descriptions and information
DISEASE_DESCRIPTIONS = {
    "Apple___Apple_scab": (
        "A fungal disease causing dark spots and cracking on leaves and fruit. "
        "Common in humid climates. Affected leaves should be removed and destroyed."
    ),
    "Apple___Black_rot": (
        "A serious fungal disease causing cankers and fruit rot. "
        "Can cause significant crop loss. Requires immediate treatment."
    ),
    "Apple___Cedar_apple_rust": (
        "A fungal disease characterized by yellow spots and fungal growths. "
        "Spreads between apple and juniper/cedar trees. "
        "Management includes removing infected juniper trees nearby."
    ),
    "Apple___healthy": (
        "The leaf is healthy with no visible signs of disease. "
        "Continue regular monitoring for disease development."
    )
}

# Disease treatment recommendations (optional)
DISEASE_RECOMMENDATIONS = {
    "Apple___Apple_scab": [
        "Remove and destroy infected leaves",
        "Apply fungicide during growing season",
        "Improve air circulation through pruning",
        "Avoid overhead watering"
    ],
    "Apple___Black_rot": [
        "Prune affected branches immediately",
        "Apply copper-based fungicide",
        "Remove cankers if possible",
        "Contact local agricultural extension for severe cases"
    ],
    "Apple___Cedar_apple_rust": [
        "Remove infected leaves",
        "Eliminate nearby juniper/cedar trees if possible",
        "Apply sulfur or fungicide",
        "Monitor for reoccurrence"
    ],
    "Apple___healthy": [
        "Continue regular monitoring",
        "Maintain good tree health",
        "Apply preventive fungicide if recommended",
        "Ensure proper water and nutrient management"
    ]
}

# ====================
# CONFIDENCE INTERPRETATION
# ====================

# Define what different confidence levels mean
CONFIDENCE_LEVELS = {
    "very_high": (0.85, 1.0, "Very High - Highly Reliable"),
    "high": (0.75, 0.85, "High - Reliable (Stage 2)"),
    "moderate": (0.70, 0.75, "Moderate - Verify Recommended"),
    "low": (0.0, 0.70, "Low - Unreliable, Image Rejected")
}

def get_confidence_level(confidence):
    """Get confidence level description."""
    for level_key, (min_conf, max_conf, description) in CONFIDENCE_LEVELS.items():
        if min_conf <= confidence <= max_conf:
            return description
    return "Unknown"

# ====================
# LOGGING CONFIGURATION
# ====================

# Logging settings
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'logs/apple_leaf_prediction.log'

# ====================
# IMAGE PREPROCESSING
# ====================

# Image preprocessing settings
IMAGE_PREPROCESSING = {
    "target_size": (224, 224),
    "center_crop": True,
    "normalize": True,              # Apply EfficientNet preprocessing
    "color_mode": "rgb",
    "interpolation": "lanczos"
}

# ====================
# PERFORMANCE SETTINGS
# ====================

# Model inference settings
INFERENCE_CONFIG = {
    "batch_size": 1,
    "verbose": 0,                   # TensorFlow verbosity (0=silent, 1=progress, 2=detailed)
    "use_gpu": True,                # Try to use GPU if available
    "max_queue_size": 10,
    "workers": 1
}

# ====================
# VALIDATION RULES
# ====================

# Input validation settings
VALIDATION = {
    "min_image_size": 100,          # Minimum image dimension
    "max_image_size": 10000,        # Maximum image dimension
    "min_file_size_bytes": 1024,    # Minimum file size
    "max_file_size_bytes": 16 * 1024 * 1024,  # Maximum file size (16 MB)
}

# ====================
# UTILITY FUNCTIONS
# ====================

def get_stage1_config():
    """Get Stage 1 configuration."""
    return STAGE1_CONFIG.copy()

def get_stage2_config():
    """Get Stage 2 configuration."""
    return STAGE2_CONFIG.copy()

def get_disease_info(disease_class):
    """Get display name and description for a disease."""
    return {
        "display_name": DISEASE_DISPLAY_NAMES.get(disease_class, disease_class),
        "description": DISEASE_DESCRIPTIONS.get(disease_class, ""),
        "recommendations": DISEASE_RECOMMENDATIONS.get(disease_class, [])
    }

def validate_config():
    """Validate that all required model files exist."""
    errors = []

    if not os.path.exists(STAGE1_MODEL_PATH):
        errors.append(f"Stage 1 model not found: {STAGE1_MODEL_PATH}")

    if not os.path.exists(STAGE1_CLASS_NAMES_PATH):
        errors.append(f"Stage 1 class names not found: {STAGE1_CLASS_NAMES_PATH}")

    if not os.path.exists(STAGE2_MODEL_PATH):
        errors.append(f"Stage 2 model not found: {STAGE2_MODEL_PATH}")

    if not os.path.exists(STAGE2_CLASS_NAMES_PATH):
        errors.append(f"Stage 2 class names not found: {STAGE2_CLASS_NAMES_PATH}")

    # Create upload folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Create logs folder if it doesn't exist
    os.makedirs(os.path.dirname(LOG_FILE) or '.', exist_ok=True)

    return errors

if __name__ == "__main__":
    # Validate configuration on import
    errors = validate_config()
    if errors:
        print("Configuration Validation Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ Configuration validation passed")
        print(f"  Stage 1 Model: {STAGE1_MODEL_PATH}")
        print(f"  Stage 2 Model: {STAGE2_MODEL_PATH}")
        print(f"  Upload Folder: {UPLOAD_FOLDER}")
