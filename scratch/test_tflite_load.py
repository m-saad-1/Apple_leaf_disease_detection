import tensorflow as tf
import os

tflite_path = r'outputs/deployment/stage2_model_float32.tflite'
if not os.path.exists(tflite_path):
    print("Model not found")
    exit(1)

print("Attempt 1: model_path=str")
try:
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    print("Success 1")
except Exception as e:
    print(f"Fail 1: {e}")

print("\nAttempt 2: model_path only")
try:
    interp = tf.lite.Interpreter(tflite_path)
    print("Success 2")
except Exception as e:
    print(f"Fail 2: {e}")

print("\nAttempt 3: model_content")
try:
    with open(tflite_path, 'rb') as f:
        interp = tf.lite.Interpreter(model_content=f.read())
    print("Success 3")
except Exception as e:
    print(f"Fail 3: {e}")
