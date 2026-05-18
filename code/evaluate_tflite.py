"""
TASK 13 — Quantized Model Evaluation
Evaluates TFLite models (float32 and quantized) on the test dataset to measure accuracy drop.

How to run:
    python code/evaluate_tflite.py --tflite_path outputs/deployment/stage2_model_quantized.tflite --data_dir dataset/test
"""

import os, sys, json, logging, argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TFLiteEvaluation')

STAGE2_CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "models", "class_names.json")
TARGET_SIZE = (224, 224)

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size; d = min(w, h)
    img = img.crop(((w-d)//2, (h-d)//2, (w+d)//2, (h+d)//2))
    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(arr)

def evaluate_tflite(tflite_path, data_dir):
    with open(STAGE2_CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    
    logger.info(f"Loading TFLite model: {tflite_path}")
    try:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    except Exception:
        with open(tflite_path, 'rb') as f:
            interpreter = tf.lite.Interpreter(model_content=f.read())
    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_dtype = input_details[0]['dtype']
    is_quantized = (input_dtype == np.uint8 or input_dtype == np.int8)
    
    y_true, y_pred = [], []
    
    logger.info(f"Evaluating on data in: {data_dir}")
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"  Class dir not found: {class_dir}")
            continue
            
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        sample_files = files[:50] # Reduced for speed
        logger.info(f"  Class {class_name}: testing {len(sample_files)} images")
        
        for fn in sample_files:
            try:
                img_path = os.path.join(class_dir, fn)
                img_data = preprocess_image(img_path)
                
                if is_quantized:
                    input_scale, input_zero_point = input_details[0]['quantization']
                    if input_dtype == np.uint8:
                        img_data = (img_data / input_scale + input_zero_point).astype(np.uint8)
                    else:
                        img_data = (img_data / input_scale + input_zero_point).astype(np.int8)
                
                interpreter.set_tensor(input_details[0]['index'], img_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                y_pred.append(np.argmax(output[0]))
                y_true.append(idx)
            except Exception as e:
                logger.error(f"Failed processing {fn}: {e}")
                continue
            
    acc = accuracy_score(y_true, y_pred)
    logger.info(f"\nEvaluation Results for {os.path.basename(tflite_path)}:")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_path', required=True)
    parser.add_argument('--data_dir', required=True)
    args = parser.parse_args()
    evaluate_tflite(args.tflite_path, args.data_dir)
