import os
import argparse
import glob
from tensorflow.keras.models import load_model
import sys
import shutil

# Add parent directory to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from explainability.gradcam_disease_focused import generate_disease_focused_gradcam

def batch_generate_gradcam(model_path, images_dir, output_dir, limit=5):
    """
    Generates Grad-CAM visualizations for a batch of images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Gather images
    image_extensions = ('*.png', '*.jpg', '*.jpeg')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, '**', ext), recursive=True))
        
    if not image_paths:
        print(f"No images found in {images_dir}")
        return
        
    image_paths = image_paths[:limit]
    
    print(f"Generating Grad-CAM for {len(image_paths)} images...")
    
    for img_path in image_paths:
        try:
            print(f"Processing {os.path.basename(img_path)}...")
            
            # Generate Grad-CAM using existing project implementation
            gradcam_path = generate_disease_focused_gradcam(
                image_path=img_path,
                model=model,
                output_dir=output_dir,
                mode="blend"
            )
            
            if gradcam_path:
                print(f"Saved Grad-CAM to {gradcam_path}")
            else:
                print(f"Failed to generate Grad-CAM for {img_path}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM examples")
    parser.add_argument('--model', type=str, default='../models/leaf_model_best.keras', help="Path to trained model")
    parser.add_argument('--images', type=str, default='../dataset/test', help="Directory containing test images")
    parser.add_argument('--output', type=str, default='../outputs/gradcam', help="Output directory")
    parser.add_argument('--limit', type=int, default=5, help="Maximum number of images to process")
    
    args = parser.parse_args()
    batch_generate_gradcam(args.model, args.images, args.output, args.limit)
