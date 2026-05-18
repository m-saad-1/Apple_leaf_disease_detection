import argparse
import sys
import os
import json

# Add parent directory to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unified_classifier import predict_leaf_disease

def main():
    parser = argparse.ArgumentParser(description="Predict apple leaf disease for a single image.")
    parser.add_argument('image_path', type=str, help="Path to the image to classify")
    parser.add_argument('--no-gradcam', action='store_true', help="Disable Grad-CAM generation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
        
    print(f"Analyzing {args.image_path}...")
    
    result = predict_leaf_disease(args.image_path, enable_gradcam=not args.no_gradcam)
    
    print("\n--- Prediction Results ---")
    print(json.dumps(result, indent=2))
    
    if result.get("success"):
        if result["stage"] == 1:
            print(f"\nFinal Diagnosis: {result['category']}")
            print(f"Message: {result['message']}")
        elif result["stage"] == 2:
            print(f"\nFinal Diagnosis: {result['disease_display']}")
            print(f"Confidence: {result['confidence']*100:.1f}%")
            print(f"Description: {result['description']}")
            if result.get("gradcam_image"):
                print(f"Grad-CAM Heatmap saved to: {result['gradcam_image']}")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
