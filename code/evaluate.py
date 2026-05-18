import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(model_path, class_names_path, dataset_path, output_dir):
    """
    Evaluates the trained model on the test dataset.
    Generates confusion matrix and classification report.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading class names from {class_names_path}...")
    with open(class_names_path, 'r') as f:
        class_names_dict = json.load(f)
        # Assuming the JSON is like {"0": "Apple_Scab", "1": ...} or similar list
        if isinstance(class_names_dict, dict):
            # Sort by keys if numeric
            class_names = [class_names_dict[str(i)] for i in range(len(class_names_dict))]
        else:
            class_names = class_names_dict
            
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # EfficientNet preprocessing
    def preprocess_fn(img):
        import tensorflow as tf
        return tf.keras.applications.efficientnet.preprocess_input(img)
        
    print(f"Loading dataset from {dataset_path}...")
    datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    print("Running predictions...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Check if classes match
    generator_class_indices = test_generator.class_indices
    sorted_classes = sorted(generator_class_indices.items(), key=lambda x: x[1])
    generator_class_names = [item[0] for item in sorted_classes]
    print(f"Dataset classes: {generator_class_names}")
    
    # We use generator_class_names if they match, else we map them.
    # We'll use generator class names for labeling if available
    labels = generator_class_names if generator_class_names else class_names
    
    # 1. Classification Report
    print("Generating classification report...")
    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=labels)
    
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_str)
    print(f"Saved classification report to {report_path}")
    
    # Save as CSV as well
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(report_csv_path)
    
    # 2. Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")
    
    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.4f}")
    
    print("Evaluation pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Apple Leaf Disease Model")
    parser.add_argument('--model', type=str, default='../models/leaf_model_best.keras', help="Path to trained .keras model")
    parser.add_argument('--classes', type=str, default='../models/class_names.json', help="Path to class names JSON")
    parser.add_argument('--dataset', type=str, default='../dataset/test', help="Path to test dataset folder")
    parser.add_argument('--output', type=str, default='../outputs', help="Output directory")
    
    args = parser.parse_args()
    evaluate_model(args.model, args.classes, args.dataset, args.output)
