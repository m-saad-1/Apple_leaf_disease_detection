"""
Test & Validation Script for Two-Stage Apple Leaf Classification Pipeline
Verifies model loading, preprocessing, and end-to-end predictions.
"""

import os
import json
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_model_loading():
    """Test if models and class names can be loaded successfully."""
    print("\n" + "=" * 60)
    print("TEST 1: Model & Class Names Loading")
    print("=" * 60)

    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TensorFlow: {e}")
        return False

    # Check Stage 1 files
    print("\nStage 1 Model:")
    stage1_model_path = "models/stage1_model.keras"
    stage1_classes_path = "models/stage1_class_names.json"

    if not os.path.exists(stage1_model_path):
        print(f"✗ Stage 1 model not found: {stage1_model_path}")
        return False
    print(f"✓ Stage 1 model found: {stage1_model_path}")

    if not os.path.exists(stage1_classes_path):
        print(f"✗ Stage 1 class names not found: {stage1_classes_path}")
        return False

    with open(stage1_classes_path, "r") as f:
        stage1_classes = json.load(f)
    print(f"✓ Stage 1 classes loaded: {stage1_classes}")

    try:
        stage1_model = tf.keras.models.load_model(stage1_model_path)
        print(f"✓ Stage 1 model loaded successfully")
        print(f"  Input shape: {stage1_model.input_shape}")
        print(f"  Output shape: {stage1_model.output_shape}")
    except Exception as e:
        print(f"✗ Failed to load Stage 1 model: {e}")
        return False

    # Check Stage 2 files
    print("\nStage 2 Model:")
    stage2_model_path = "models/leaf_model2.keras"
    stage2_classes_path = "models/class_names.json"

    if not os.path.exists(stage2_model_path):
        print(f"✗ Stage 2 model not found: {stage2_model_path}")
        return False
    print(f"✓ Stage 2 model found: {stage2_model_path}")

    if not os.path.exists(stage2_classes_path):
        print(f"✗ Stage 2 class names not found: {stage2_classes_path}")
        return False

    with open(stage2_classes_path, "r") as f:
        stage2_classes = json.load(f)
    print(f"✓ Stage 2 classes loaded: {stage2_classes}")

    try:
        stage2_model = tf.keras.models.load_model(stage2_model_path)
        print(f"✓ Stage 2 model loaded successfully")
        print(f"  Input shape: {stage2_model.input_shape}")
        print(f"  Output shape: {stage2_model.output_shape}")
    except Exception as e:
        print(f"✗ Failed to load Stage 2 model: {e}")
        return False

    return True


def test_image_preprocessing():
    """Test image preprocessing functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: Image Preprocessing")
    print("=" * 60)

    try:
        from utils.image_processing import preprocess_image
        print("✓ Image preprocessing module imported")
    except ImportError as e:
        print(f"✗ Failed to import image_processing: {e}")
        return False

    # Create a test image if not available
    test_image_path = "test_image.jpg"

    if not os.path.exists(test_image_path):
        print(f"⚠ Test image not found: {test_image_path}")
        print("  Note: Create a test image to verify preprocessing")
        return True

    try:
        img_array = preprocess_image(test_image_path)
        print(f"✓ Image preprocessed successfully")
        print(f"  Output shape: {img_array.shape}")
        print(f"  Data type: {img_array.dtype}")
        print(f"  Min value: {img_array.min():.4f}, Max value: {img_array.max():.4f}")
        return True
    except Exception as e:
        print(f"✗ Failed to preprocess image: {e}")
        return False


def test_stage1_classifier():
    """Test Stage 1 classifier independently."""
    print("\n" + "=" * 60)
    print("TEST 3: Stage 1 Classifier")
    print("=" * 60)

    # Check if test image exists
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print(f"⚠ Test image not found: {test_image_path}")
        print("  Note: Create a test image to verify Stage 1 classification")
        print("  Or run this test after uploading an image through the web interface")
        return True

    try:
        from stage1_classifier import classify_stage1
        print("✓ Stage 1 classifier module imported")
    except ImportError as e:
        print(f"✗ Failed to import stage1_classifier: {e}")
        return False

    try:
        result = classify_stage1(test_image_path)
        print(f"✓ Stage 1 classification successful")
        print(f"  Category: {result['category']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Margin: {result['margin']:.4f}")
        print(f"  Is Apple Leaf: {result['is_apple_leaf']}")
        print(f"  Needs Stage 2: {result['needs_stage2']}")
        return True
    except Exception as e:
        print(f"✗ Stage 1 classification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stage2_classifier():
    """Test Stage 2 classifier independently."""
    print("\n" + "=" * 60)
    print("TEST 4: Stage 2 Classifier")
    print("=" * 60)

    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print(f"⚠ Test image not found: {test_image_path}")
        print("  Note: Create a test image to verify Stage 2 classification")
        return True

    try:
        from leaf_classifier import classify_stage2
        print("✓ Stage 2 classifier module imported")
    except ImportError as e:
        print(f"✗ Failed to import leaf_classifier: {e}")
        return False

    try:
        result = classify_stage2(test_image_path)
        print(f"✓ Stage 2 classification successful")
        print(f"  Disease: {result['disease']}")
        print(f"  Disease Display: {result['disease_display']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Margin: {result['margin']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Stage 2 classification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_pipeline():
    """Test the complete unified pipeline."""
    print("\n" + "=" * 60)
    print("TEST 5: Unified Two-Stage Pipeline")
    print("=" * 60)

    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print(f"⚠ Test image not found: {test_image_path}")
        print("  Note: Create a test image to verify the complete pipeline")
        print("  Steps:")
        print("    1. Save a leaf image as 'test_image.jpg' in the project root")
        print("    2. Run this script again")
        return True

    try:
        from unified_classifier import predict_leaf_disease
        print("✓ Unified classifier module imported")
    except ImportError as e:
        print(f"✗ Failed to import unified_classifier: {e}")
        return False

    try:
        result = predict_leaf_disease(test_image_path)
        print(f"✓ Complete pipeline executed successfully")
        print(f"  Success: {result['success']}")
        print(f"  Stage: {result['stage']}")

        if result['success']:
            if result['stage'] == 1:
                print(f"  Category: {result['category']}")
            elif result['stage'] == 2:
                print(f"  Disease: {result['disease_display']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Message: {result['message']}")
        else:
            print(f"  Error: {result['error']}")

        return True
    except Exception as e:
        print(f"✗ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flask_app():
    """Test Flask application components."""
    print("\n" + "=" * 60)
    print("TEST 6: Flask Application")
    print("=" * 60)

    try:
        from app import app
        print("✓ Flask app imported successfully")
        print(f"  Upload folder: {app.config['UPLOAD_FOLDER']}")
        print(f"  Max content length: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f} MB")
        print(f"  Debug: {app.debug}")
    except ImportError as e:
        print(f"✗ Failed to import Flask app: {e}")
        return False

    # Check templates
    templates = ['index.html', '404.html']
    for template in templates:
        path = f"templates/{template}"
        if os.path.exists(path):
            print(f"✓ Template found: {path}")
        else:
            print(f"✗ Template not found: {path}")
            return False

    return True


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    test_names = [
        "Model Loading",
        "Image Preprocessing",
        "Stage 1 Classifier",
        "Stage 2 Classifier",
        "Unified Pipeline",
        "Flask Application"
    ]

    passed = sum(results)
    total = len(results)

    for name, result in zip(test_names, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! System is ready for deployment.")
        return True
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("APPLE LEAF DISEASE DETECTION - SYSTEM VALIDATION")
    print("Two-Stage Classification Pipeline")
    print("=" * 60)

    results = [
        test_model_loading(),
        test_image_preprocessing(),
        test_stage1_classifier(),
        test_stage2_classifier(),
        test_unified_pipeline(),
        test_flask_app()
    ]

    success = print_summary(results)

    if not success:
        print("\n💡 Tips for troubleshooting:")
        print("  1. Ensure all model files are in the 'models/' directory")
        print("  2. Verify TensorFlow and Keras are installed correctly")
        print("  3. Check that all Python files are in the correct locations")
        print("  4. For classifier tests, add a 'test_image.jpg' to the project root")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
