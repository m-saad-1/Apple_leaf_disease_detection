import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import glob

def apply_distortion(img, distortion_type, severity):
    """
    Applies a specific distortion at a given severity to a PIL Image.
    """
    if distortion_type == "noise":
        # severity is std dev of Gaussian noise
        img_array = np.array(img)
        noise = np.random.normal(0, severity, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
        
    elif distortion_type == "blur":
        # severity is blur radius
        return img.filter(ImageFilter.GaussianBlur(radius=severity))
        
    elif distortion_type == "brightness":
        # severity is brightness factor (1.0 is original)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(severity)
        
    return img

def evaluate_robustness(model_path, dataset_dir, output_dir):
    """
    Evaluates model robustness against synthetic distortions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # EfficientNet preprocessing
    import tensorflow as tf
    def preprocess(img):
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Get test images and true labels
    # Assumes dataset_dir has subfolders for classes
    classes = sorted(os.listdir(dataset_dir))
    image_paths = []
    y_true = []
    
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls)
        if os.path.isdir(cls_dir):
            paths = glob.glob(os.path.join(cls_dir, '*.jpg'))[:20] # Limit per class to speed up test
            image_paths.extend(paths)
            y_true.extend([i] * len(paths))
            
    if not image_paths:
        print(f"No images found in {dataset_dir} subdirectories.")
        print("Please structure dataset as: dataset_dir/class_name/image.jpg")
        return

    print(f"Evaluating {len(image_paths)} images...")

    # Define distortions and severities
    distortions = {
        "noise": [0, 10, 25, 50, 75], # std dev
        "blur": [0, 1, 3, 5, 9], # radius
        "brightness": [1.0, 0.8, 0.6, 0.4, 0.2, 1.2, 1.4, 1.6] # factor
    }

    results = []

    for dist_type, severities in distortions.items():
        print(f"\nTesting {dist_type} robustness...")
        accuracies = []
        
        for sev in severities:
            y_pred = []
            for img_path in image_paths:
                img = Image.open(img_path).convert('RGB')
                distorted_img = apply_distortion(img, dist_type, sev)
                processed_img = preprocess(distorted_img)
                
                pred = model.predict(processed_img, verbose=0)
                y_pred.append(np.argmax(pred[0]))
                
            acc = accuracy_score(y_true, y_pred)
            accuracies.append(acc)
            print(f"  Severity {sev}: Accuracy = {acc:.4f}")
            
            results.append({
                'Distortion': dist_type,
                'Severity': sev,
                'Accuracy': acc
            })

        # Plot for this distortion
        plt.figure(figsize=(8, 5))
        plt.plot(severities, accuracies, marker='o', linestyle='-', linewidth=2)
        plt.title(f'Robustness against {dist_type.capitalize()}')
        plt.xlabel('Severity')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plot_path = os.path.join(output_dir, f'robustness_{dist_type}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()

    # Save all results
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'robustness_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved robustness results to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model robustness")
    parser.add_argument('--model', type=str, default='../models/leaf_model_best.keras', help="Path to trained model")
    parser.add_argument('--dataset', type=str, default='../dataset/test', help="Path to test dataset folder")
    parser.add_argument('--output', type=str, default='../outputs/robustness', help="Output directory")
    
    args = parser.parse_args()
    evaluate_robustness(args.model, args.dataset, args.output)
