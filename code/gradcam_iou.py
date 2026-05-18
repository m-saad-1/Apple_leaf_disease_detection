import os
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from explainability.gradcam_disease_focused import make_gradcam_heatmap
from tensorflow.keras.models import load_model

def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(array)

def compute_iou(mask1, mask2):
    """
    Computes Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0:
        return 0.0
    return np.sum(intersection) / np.sum(union)

def binarize_heatmap(heatmap, threshold=0.5):
    """
    Converts a continuous heatmap (0-1) to a binary mask based on a threshold.
    """
    return (heatmap > threshold).astype(np.uint8)

def evaluate_gradcam_iou(model_path, images_dir, annotations_dir, output_dir, threshold=0.5):
    """
    Evaluates Grad-CAM localization using Intersection over Union (IoU) against ground truth masks.
    """
    import tensorflow as tf
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Try to find the last conv layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4 and isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
            
    if not last_conv_layer_name:
        # Generic fallback
        last_conv_layer_name = "top_conv" 
    
    results = []
    
    print("""
    ========================================================================
    ANNOTATION WORKFLOW EXPLANATION:
    1. Ground Truth Creation: Use tools like LabelImg (for bounding boxes) 
       or CVAT (for polygon masks) to label the diseased areas.
    2. Mask Generation: Convert the XML/JSON annotations into binary mask images 
       (where 255=disease, 0=background) matching the original image dimensions.
    3. Evaluation: This script compares the binarized Grad-CAM heatmap against 
       the ground truth binary mask using the Intersection over Union (IoU) metric.
    ========================================================================
    """)
    
    # Check if annotations directory exists and has files
    if not os.path.exists(annotations_dir) or not os.listdir(annotations_dir):
        print(f"Warning: No ground truth masks found in {annotations_dir}.")
        print("Running in DEMO mode with synthetic masks...")
        demo_mode = True
    else:
        demo_mode = False

    import glob
    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))[:5] # Limit to 5 for demo
    
    if not image_paths:
        print(f"No images found in {images_dir}")
        return
        
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        print(f"Evaluating {img_name}...")
        
        # 1. Get Grad-CAM Heatmap
        img_array = get_img_array(img_path, size=(224, 224))
        
        try:
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            heatmap = cv2.resize(heatmap, (224, 224))
            pred_mask = binarize_heatmap(heatmap, threshold)
        except Exception as e:
            print(f"Failed to generate Grad-CAM for {img_name}: {e}")
            continue
            
        # 2. Get Ground Truth Mask
        mask_name = img_name.replace('.jpg', '_mask.png')
        mask_path = os.path.join(annotations_dir, mask_name)
        
        if demo_mode or not os.path.exists(mask_path):
            # Create a synthetic circular mask in the center for demo purposes
            gt_mask = np.zeros((224, 224), dtype=np.uint8)
            cv2.circle(gt_mask, (112, 112), 40, 1, -1)
        else:
            gt_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.resize(gt_img, (224, 224))
            gt_mask = (gt_img > 127).astype(np.uint8)
            
        # 3. Compute IoU
        iou_score = compute_iou(gt_mask, pred_mask)
        results.append({
            'Image': img_name,
            'IoU': iou_score
        })
        
        # 4. Generate Visual Comparison
        original_img = cv2.imread(img_path)
        original_img = cv2.resize(original_img, (224, 224))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title(f'Binarized Grad-CAM (t={threshold})')
        axes[2].axis('off')
        
        axes[3].imshow(gt_mask, cmap='gray')
        axes[3].set_title('Ground Truth Mask')
        axes[3].axis('off')
        
        plt.tight_layout()
        vis_path = os.path.join(output_dir, f'iou_vis_{img_name}')
        plt.savefig(vis_path, dpi=150)
        plt.close()

    if results:
        df = pd.DataFrame(results)
        mean_iou = df['IoU'].mean()
        
        print("\n--- IoU Results ---")
        print(df.to_string(index=False))
        print(f"\nMean IoU: {mean_iou:.4f}")
        
        csv_path = os.path.join(output_dir, 'gradcam_iou_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Grad-CAM using IoU")
    parser.add_argument('--model', type=str, default='../models/leaf_model_best.keras', help="Path to trained model")
    parser.add_argument('--images', type=str, default='../data/test_images', help="Directory with test images")
    parser.add_argument('--annotations', type=str, default='../annotations/masks', help="Directory with ground truth masks")
    parser.add_argument('--output', type=str, default='../outputs/gradcam', help="Output directory")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for binarizing heatmap")
    
    args = parser.parse_args()
    evaluate_gradcam_iou(args.model, args.images, args.annotations, args.output, args.threshold)
