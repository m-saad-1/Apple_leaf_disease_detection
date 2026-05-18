# Apple Leaf Disease Detection System

## Overview

This project is a Flask-based deep learning application for detecting diseases in apple leaf images. It uses a two-stage classification pipeline to first verify whether an image is an apple leaf, then identify the specific disease if the leaf is diseased. The system also generates Grad-CAM visual explanations so users can see which regions influenced the prediction.

## What the system does

- Accepts an uploaded leaf image through a web UI.
- Checks whether the image is an apple leaf.
- Short-circuits if the leaf is healthy or rejects the image if it is not an apple leaf.
- Runs a second classifier only when the leaf is diseased.
- Returns the disease class, confidence values, and an optional Grad-CAM heatmap.

## Dataset

The repository does not include the training dataset itself, but the class files and model outputs show the dataset schema used for training.

### Stage 1 dataset labels

The first model is trained on three categories:

- Apple_Diseased
- Apple_Healthy
- Not_Apple_Leaf

This stage appears designed to filter out non-apple images and separate healthy from diseased apple leaves.

### Stage 2 dataset labels

The second model is trained on four disease classes:

- Apple___Apple_scab
- Apple___Black_rot
- Apple___Cedar_apple_rust
- Apple___healthy

These labels match a standard apple leaf disease classification task where the model predicts a specific disease or healthy leaf status.

### Dataset-driven behavior

- Stage 1 acts as a gatekeeper for input validity.
- Stage 2 specializes in disease-level classification.
- The class-name JSON files in the models folder confirm the exact label order used at inference time.

## Models

### Stage 1 model

- File: `models/stage1_model.keras`
- Class names: `models/stage1_class_names.json`
- Purpose: apple leaf detection and coarse health triage
- Input size: 224 x 224 RGB
- Output: 3-class softmax prediction
- Architecture: EfficientNet-based transfer learning, as described in the project documentation and preprocessing pipeline

### Stage 2 model

- File: `models/leaf_model2.keras`
- Class names: `models/class_names.json`
- Purpose: disease classification for apple leaves
- Input size: 224 x 224 RGB
- Output: 4-class softmax prediction
- Architecture: EfficientNet-based model, likely fine-tuned for apple disease recognition

### Inference design

The system loads both models at startup and uses confidence and margin thresholds to decide whether a prediction is strong enough to return.

## Framework and libraries

The project uses the following stack:

- Python 3.8+
- TensorFlow 2.14.0
- Keras 2.14.0
- Flask 2.3.2
- OpenCV 4.7.0.72
- Pillow 10.0.0
- NumPy 1.26.4
- Gunicorn 21.2.0 for production serving
- python-dotenv 1.0.0 for environment configuration support

## Application structure

### Main files

- `app.py`: Flask application entry point and HTTP routes
- `unified_classifier.py`: orchestrates the two-stage prediction flow
- `stage1_classifier.py`: apple leaf detection model inference
- `leaf_classifier.py`: disease classification model inference
- `predict.py`: command-line prediction helper
- `config.py`: centralized configuration, paths, thresholds, and disease metadata

### Supporting folders

- `templates/`: HTML pages for the web interface
- `static/css/`: UI styling
- `static/uploads/`: uploaded images stored temporarily
- `static/gradcam/`: generated explainability outputs
- `utils/`: image preprocessing helpers
- `explainability/`: Grad-CAM implementations
- `models/`: trained model files and class metadata

## Image preprocessing

All model inputs go through a shared preprocessing pipeline in `utils/image_processing.py`.

### Steps

1. Load the image with Pillow.
2. Convert to RGB.
3. Optionally center-crop to square.
4. Resize to 224 x 224.
5. Convert to a NumPy array.
6. Add a batch dimension.
7. Apply EfficientNet preprocessing, which normalizes pixel values to the expected range.

### Why this matters

- Keeps Stage 1 and Stage 2 preprocessing consistent.
- Reduces the effect of noisy backgrounds.
- Matches the input shape expected by the EfficientNet-based classifiers.

## Explainability

The project includes Grad-CAM to make predictions interpretable.

### Available implementations

- `explainability/gradcam.py`: general Grad-CAM implementation with automatic last-convolution-layer detection.
- `explainability/gradcam_disease_focused.py`: the primary explainability module, tuned to suppress leaf-edge artifacts and emphasize disease regions.
- `explainability/gradcam_enhanced.py`: compatibility wrapper that forwards to the disease-focused implementation.
- `explainability/gradcam_simple.py`: simpler alternative implementation.

### What the explainability module does

- Locates the last convolutional layer automatically.
- Computes gradient-based activation maps.
- Suppresses edge regions that often correspond to leaf borders rather than disease.
- Produces a heatmap overlay saved under `static/gradcam/`.

## Evaluation and testing

The repository contains test scripts that validate the pipeline end to end.

### Test files

- `test_system.py`: checks model loading, preprocessing, and prediction flow.
- `test_gradcam_direct.py`: validates Grad-CAM generation directly.
- `test_gradcam_ui.py`: checks the UI integration for Grad-CAM output.

### What evaluation should cover

- Model loading success for both stages.
- Correct class-name mapping.
- Preprocessing consistency.
- Stage 1 classification correctness.
- Stage 2 disease classification correctness.
- Confidence and margin threshold behavior.
- Grad-CAM generation and visual quality.

### Runtime thresholds

The configuration file sets the default decision thresholds:

- Stage 1 confidence threshold: 0.70
- Stage 1 margin threshold: 0.15
- Stage 2 confidence threshold: 0.75
- Stage 2 margin threshold: 0.20

These thresholds help avoid weak or ambiguous predictions.

## Configuration

`config.py` centralizes the most important runtime settings.

### Key settings

- Upload folder: `static/uploads`
- Allowed file types: `png`, `jpg`, `jpeg`
- Max upload size: 16 MB
- Flask host: `0.0.0.0`
- Flask port: `5001`
- Debug mode: enabled

### Disease metadata

The configuration file also includes:

- Human-readable disease display names
- Disease descriptions
- Treatment recommendations

This makes the output more useful than a raw class prediction.

## Web interface

The UI is rendered with Flask templates and styled using `static/css/style.css`.

### User flow

1. Open the homepage.
2. Upload an apple leaf image.
3. Preview the image.
4. Run prediction.
5. Review Stage 1 result.
6. If diseased, review Stage 2 result and Grad-CAM heatmap.

### Frontend elements

- Upload area with drag-and-drop support
- Preview panel
- Prediction button
- Result sections for Stage 1 and Stage 2
- Final diagnosis summary

## Prediction flow

The decision logic in `unified_classifier.py` works like this:

1. Validate that the image exists.
2. Run Stage 1.
3. If the image is not an apple leaf, stop and return a rejection.
4. If the leaf is healthy, return a healthy result immediately.
5. If the leaf is diseased, run Stage 2.
6. Optionally generate Grad-CAM for the disease result.

## Deployment and usage

### Local run

- Create and activate a virtual environment.
- Install dependencies from `requirements.txt`.
- Start the app with `python app.py`.

### CLI usage

`python predict.py <image_path>`

### Production considerations

- Gunicorn is included for deployment.
- The app is stateless apart from uploaded files and generated Grad-CAM images.
- Model loading happens at startup, so deployment should ensure the `.keras` files are present.

## Strengths of the project

- Two-stage design improves robustness.
- Transfer-learning-based models reduce training cost.
- Centralized configuration makes tuning easier.
- Grad-CAM improves trust and interpretability.
- The project includes both web and CLI interfaces.
- Validation scripts support testing and debugging.

## Notes and limitations

- The training dataset itself is not stored in the repository.
- Exact training metrics such as accuracy, precision, recall, and F1 score are not included in the visible files.
- The repository appears optimized for inference and presentation rather than full training reproducibility.

## Short summary

This is a well-structured apple leaf disease detection project built with Flask and TensorFlow. It uses EfficientNet-based classifiers in a two-stage pipeline, supports image preprocessing and Grad-CAM explainability, and includes configuration, testing, and deployment support.