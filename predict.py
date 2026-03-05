"""
Prediction Module: Two-Stage Apple Leaf Disease Classification Pipeline

This module provides the main prediction interface for the apple leaf disease detection system.
It uses a unified two-stage classification approach:
- Stage 1: Detects if the image is an apple leaf and its health status
- Stage 2: Identifies the specific disease type if a diseased leaf is detected in Stage 1
"""

from unified_classifier import predict_leaf_disease


if __name__ == "__main__":
    # Example: Quick test
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        result = predict_leaf_disease(test_image)
        print("Prediction Result:")
        print(result)
    else:
        print("Usage: python predict.py <image_path>")
    if os.path.exists(test_image):
        res = predict_leaf_disease(test_image)
        print("Prediction Result:")
        for k, v in res.items():
            print(f"{k}: {v}")
    else:
        print(f"Test image '{test_image}' not found.")