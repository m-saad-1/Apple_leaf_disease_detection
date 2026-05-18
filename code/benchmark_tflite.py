import os
import argparse
import time
import numpy as np
import tensorflow as tf

def benchmark_models(keras_model_path, tflite_model_path, output_dir, num_runs=100):
    """
    Benchmarks inference latency for both Keras and TFLite models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Benchmarking over {num_runs} runs per model...")
    
    # Generate dummy input data (matches EfficientNet input shape)
    input_shape = (1, 224, 224, 3)
    dummy_input = np.random.random(input_shape).astype(np.float32)
    
    # 1. Keras Benchmark
    print(f"\nLoading Keras model: {keras_model_path}")
    try:
        keras_model = tf.keras.models.load_model(keras_model_path)
        
        # Warmup
        print("Warming up Keras model...")
        for _ in range(10):
            _ = keras_model.predict(dummy_input, verbose=0)
            
        print("Running Keras benchmark...")
        keras_times = []
        for _ in range(num_runs):
            start = time.time()
            _ = keras_model.predict(dummy_input, verbose=0)
            keras_times.append(time.time() - start)
            
        keras_avg = np.mean(keras_times) * 1000 # in ms
        keras_std = np.std(keras_times) * 1000
        print(f"Keras Latency: {keras_avg:.2f} ms ± {keras_std:.2f} ms")
        
    except Exception as e:
        print(f"Failed to benchmark Keras model: {e}")
        keras_avg = -1

    # 2. TFLite Benchmark
    print(f"\nLoading TFLite model: {tflite_model_path}")
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Warmup
        print("Warming up TFLite model...")
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            
        print("Running TFLite benchmark...")
        tflite_times = []
        for _ in range(num_runs):
            start = time.time()
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            tflite_times.append(time.time() - start)
            
        tflite_avg = np.mean(tflite_times) * 1000 # in ms
        tflite_std = np.std(tflite_times) * 1000
        print(f"TFLite Latency: {tflite_avg:.2f} ms ± {tflite_std:.2f} ms")
        
    except Exception as e:
        print(f"Failed to benchmark TFLite model: {e}")
        tflite_avg = -1

    # Save results
    if keras_avg > 0 and tflite_avg > 0:
        speedup = keras_avg / tflite_avg
        print(f"\nTFLite Speedup: {speedup:.2f}x")
        
        results_path = os.path.join(output_dir, 'latency_benchmark.txt')
        with open(results_path, 'w') as f:
            f.write(f"Inference Latency Benchmark ({num_runs} runs)\n")
            f.write("========================================\n")
            f.write(f"Hardware/Environment: {tf.config.list_physical_devices()}\n\n")
            f.write(f"Keras (.keras):  {keras_avg:.2f} ms (±{keras_std:.2f})\n")
            f.write(f"TFLite (.tflite): {tflite_avg:.2f} ms (±{tflite_std:.2f})\n")
            f.write(f"Speedup:          {speedup:.2f}x\n")
        print(f"Results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark inference latency")
    parser.add_argument('--keras_model', type=str, default='../models/leaf_model_best.keras', help="Path to Keras model")
    parser.add_argument('--tflite_model', type=str, default='../outputs/deployment/model_dynamic_quant.tflite', help="Path to TFLite model")
    parser.add_argument('--output', type=str, default='../outputs/deployment', help="Output directory")
    parser.add_argument('--runs', type=int, default=100, help="Number of benchmark runs")
    
    args = parser.parse_args()
    benchmark_models(args.keras_model, args.tflite_model, args.output, args.runs)
