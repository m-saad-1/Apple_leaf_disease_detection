import os
import json
import numpy as np
import sys

# Suppress TensorFlow info/warning messages for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
except ImportError:
    print("Error: TensorFlow is required. Please install it using 'pip install tensorflow'.")
    sys.exit(1)

class Stage1Predictor:
    def __init__(self, model_path, class_names_path):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names = None
        
        self._load_assets()

    def _load_assets(self):
        """Loads the model and class names from the specified paths."""
        print(f"Loading class names from: {self.class_names_path}")
        try:
            with open(self.class_names_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"Loaded {len(self.class_names)} classes.")
        except Exception as e:
            print(f"Failed to load class names: {e}")
            raise

        print(f"Loading model from: {self.model_path}")
        try:
            # compile=False is safer for inference if custom optimizers were used during training
            self.model = load_model(self.model_path, compile=False)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def predict_image(self, image_path):
        """
        Predicts the class of the image provided at image_path.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # 1. Determine Input Shape
        # Try to get input shape from model, default to 224x224 if not available
        try:
            input_shape = self.model.input_shape[1:3]
            if input_shape[0] is None: 
                input_shape = (224, 224)
        except AttributeError:
            input_shape = (224, 224)

        # 2. Load and Preprocess Image
        img = load_img(image_path, target_size=input_shape)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize pixel values to [0, 1] (Standard for this type of model)
        img_array = img_array / 255.0
        
        # 3. Prediction
        predictions = self.model.predict(img_array, verbose=0)
        
        # 4. Process Results
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index]
        predicted_class = self.class_names[predicted_index]
        
        return predicted_class, float(confidence)

if __name__ == "__main__":
    # Define paths based on your project structure
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_FILE = os.path.join(BASE_DIR, 'models', 'stage1_apple_classifier.h5')
    CLASSES_FILE = os.path.join(BASE_DIR, 'models', 'stage1_class_names.json')
    
    # Initialize Predictor
    try:
        predictor = Stage1Predictor(MODEL_FILE, CLASSES_FILE)
        
        # Check for image argument
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            label, conf = predictor.predict_image(image_path)
            print(f"\nPrediction: {label}")
            print(f"Confidence: {conf:.2%}")
        else:
            print("\nUsage: python stage1_prediction.py <path_to_image>")
    except Exception as e:
        print(f"Initialization Error: {e}")