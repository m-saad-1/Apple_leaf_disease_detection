import os
import argparse
import tensorflow as tf

def convert_to_tflite(model_path, output_dir):
    """
    Converts a Keras model to TensorFlow Lite formats (standard and quantized).
    Compares the file sizes to demonstrate compression.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Original size
    original_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Original Model Size (.keras): {original_size_mb:.2f} MB")

    # 1. Standard TFLite (Float32)
    print("Converting to standard TFLite (Float32)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(output_dir, 'model_float32.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    float_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"TFLite Float32 Size: {float_size_mb:.2f} MB (Compression: {original_size_mb/float_size_mb:.2f}x)")

    # 2. Dynamic Range Quantization (Int8 weights, Float32 activations)
    print("Converting to Dynamic Range Quantized TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    
    tflite_quant_path = os.path.join(output_dir, 'model_dynamic_quant.tflite')
    with open(tflite_quant_path, 'wb') as f:
        f.write(tflite_quant_model)
        
    quant_size_mb = os.path.getsize(tflite_quant_path) / (1024 * 1024)
    print(f"TFLite Dynamic Quant Size: {quant_size_mb:.2f} MB (Compression: {original_size_mb/quant_size_mb:.2f}x)")

    # Write summary to file
    summary_path = os.path.join(output_dir, 'conversion_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("TFLite Conversion Summary\n")
        f.write("=========================\n")
        f.write(f"Original (.keras):     {original_size_mb:.2f} MB\n")
        f.write(f"TFLite (Float32):      {float_size_mb:.2f} MB\n")
        f.write(f"TFLite (Dynamic Quant): {quant_size_mb:.2f} MB\n")
    
    print(f"\nConversion complete. Outputs saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras model to TFLite")
    parser.add_argument('--model', type=str, default='../models/leaf_model_best.keras', help="Path to trained model")
    parser.add_argument('--output', type=str, default='../outputs/deployment', help="Output directory")
    
    args = parser.parse_args()
    convert_to_tflite(args.model, args.output)
