"""
Unified Two-Stage Classification Pipeline
Orchestrates both Stage 1 (apple leaf detection) and Stage 2 (disease classification)
for a complete apple leaf disease detection system.

Includes Grad-CAM explainability for Stage 2 disease predictions.
"""

from stage1_classifier import classify_stage1
from leaf_classifier import classify_stage2, stage2_model
import tensorflow as tf
import os


def predict_leaf_disease(img_path, enable_gradcam=True):
    """
    Complete two-stage prediction pipeline for apple leaf disease detection.

    Workflow:
    1. Stage 1: Classify as "Apple_Healthy", "Apple_Diseased", or "Not_Apple_Leaf"
    2. If "Not_Apple_Leaf" → Return rejection (not an apple leaf)
    3. If "Apple_Healthy" → Return healthy result (no disease)
    4. If "Apple_Diseased" → Forward to Stage 2 for specific disease identification
    5. Generate Grad-CAM explainability for Stage 2 predictions (if enabled)

    Args:
        img_path (str): Path to the image file to classify
        enable_gradcam (bool): Whether to generate Grad-CAM visualization for Stage 2 predictions (default: True)

    Returns:
        dict: Comprehensive prediction result with unified structure.
              Stage 2 results include 'gradcam_image' field with path to explainability visualization.
    """

    # Validate image file exists
    if not os.path.exists(img_path):
        return {
            "success": False,
            "error": f"Image file not found: {img_path}",
            "stage": 0,
            "category": "Error",
            "disease": None,
            "confidence": 0.0
        }

    # ====================
    # STAGE 1: Apple Leaf Detection
    # ====================
    stage1_result = classify_stage1(img_path)

    # If error in Stage 1
    if stage1_result["category"] == "Error":
        return {
            "success": False,
            "error": stage1_result["message"],
            "stage": 1,
            "category": "Error",
            "disease": None,
            "confidence": 0.0
        }

    # If not an apple leaf - return as a valid detection outcome
    if stage1_result["category"] == "Not_Apple_Leaf":
        return {
            "success": True,
            "error": None,
            "stage": 1,
            "category": "Not_Apple_Leaf",
            "disease": None,
            "confidence": stage1_result["confidence"],
            "message": stage1_result["message"],
            "details": {
                "stage1": {
                    "category": stage1_result["category"],
                    "confidence": stage1_result["confidence"],
                    "margin": stage1_result["margin"]
                }
            }
        }

    # If apple leaf is healthy - RETURN HEALTHY
    if stage1_result["category"] == "Apple_Healthy":
        return {
            "success": True,
            "error": None,
            "stage": 1,
            "category": "Healthy",
            "disease": "Apple___healthy",
            "disease_display": "Healthy",
            "confidence": stage1_result["confidence"],
            "message": "This apple leaf is healthy and shows no signs of disease.",
            "details": {
                "stage1": {
                    "category": stage1_result["category"],
                    "confidence": stage1_result["confidence"],
                    "margin": stage1_result["margin"]
                }
            }
        }

    # ====================
    # STAGE 2: Disease Classification (for Apple_Diseased)
    # ====================
    if stage1_result["category"] == "Apple_Diseased":
        stage2_result = classify_stage2(img_path)

        # If error in Stage 2
        if stage2_result["disease"] == "Error":
            return {
                "success": False,
                "error": stage2_result["message"],
                "stage": 2,
                "category": "Apple_Diseased",
                "disease": None,
                "confidence": 0.0,
                "details": {
                    "stage1": {
                        "category": stage1_result["category"],
                        "confidence": stage1_result["confidence"],
                        "margin": stage1_result["margin"]
                    }
                }
            }

        # ====================
        # GRAD-CAM EXPLAINABILITY (Stage 2 only)
        # ====================
        gradcam_image = None
        gradcam_error = None
        
        if enable_gradcam:
            try:
                # Use disease-focused Grad-CAM for accurate disease localization
                from explainability import generate_disease_focused_gradcam
                
                # Generate with "blend" mode for good presentation
                gradcam_path = generate_disease_focused_gradcam(
                    image_path=img_path,
                    model=stage2_model,
                    output_dir="static/gradcam",
                    mode="blend"  # blend, contour, or highlight
                )
                
                if gradcam_path:
                    gradcam_image = gradcam_path
                else:
                    gradcam_error = "Grad-CAM generation returned no path"
                    
            except Exception as e:
                # Grad-CAM failure should not break prediction pipeline
                gradcam_error = f"Grad-CAM generation failed: {str(e)}"
                import traceback
                traceback.print_exc()

        # Return Stage 2 classification with Grad-CAM
        result = {
            "success": True,
            "error": None,
            "stage": 2,
            "category": "Apple_Diseased",
            "disease": stage2_result["disease"],
            "disease_display": stage2_result["disease_display"],
            "confidence": stage2_result["confidence"],
            "description": stage2_result["description"],
            "message": stage2_result["message"],
            "gradcam_image": gradcam_image,  # Path to Grad-CAM visualization
            "details": {
                "stage1": {
                    "category": stage1_result["category"],
                    "confidence": stage1_result["confidence"],
                    "margin": stage1_result["margin"]
                },
                "stage2": {
                    "disease": stage2_result["disease"],
                    "disease_display": stage2_result["disease_display"],
                    "confidence": stage2_result["confidence"],
                    "margin": stage2_result["margin"],
                    "description": stage2_result["description"]
                }
            }
        }

        # Propagate Stage 1 reliability warning when Stage 1 confidence is low.
        if not stage1_result.get("needs_stage2", False) and stage1_result.get("message"):
            result["details"]["stage1_warning"] = stage1_result["message"]
            result["message"] = (
                f"{result['message']} Note: Stage 1 had lower confidence; interpret with caution."
            )
        
        # Add Grad-CAM error to details if it failed (for debugging)
        if gradcam_error:
            result["details"]["gradcam_error"] = gradcam_error
        
        return result

    # Low-confidence Stage 1 or unexpected category fallback
    return {
        "success": False,
        "error": stage1_result.get("message", "Unexpected classification result"),
        "stage": 1,
        "category": stage1_result["category"],
        "disease": None,
        "confidence": stage1_result.get("confidence", 0.0),
        "details": {
            "stage1": {
                "category": stage1_result.get("category"),
                "confidence": stage1_result.get("confidence", 0.0),
                "margin": stage1_result.get("margin", 0.0)
            }
        }
    }
