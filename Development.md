You are an expert AI research engineer and senior machine learning systems developer. Your task is to help me COMPLETE and FINALIZE my existing AI research project professionally and systematically.

IMPORTANT CONTEXT:
- The project is ALREADY BUILT.
- The trained models ALREADY EXIST.
- I am NOT rebuilding from scratch.
- I am NOT retraining unless absolutely necessary.
- The project is now in the FINAL EVALUATION + OUTPUT GENERATION stage.
- We are working LOCALLY in VS Code, NOT primarily in Google Colab.
- Focus ONLY on implementation completion, evaluation, experiments, outputs, deployment, and code organization.
- Do NOT rewrite the research paper yet unless explicitly asked later.

========================================================
PROJECT DETAILS
========================================================

PROJECT TITLE:
Explainable and Robust AI System for Apple Leaf Disease Detection Using Deep Learning

CURRENT SYSTEM:
- Flask-based deep learning web application
- Two-stage classification pipeline
- TensorFlow/Keras
- EfficientNet-based models
- Grad-CAM explainability
- Saved .keras models already available

STAGE 1 MODEL:
- Detects:
  - Apple_Diseased
  - Apple_Healthy
  - Not_Apple_Leaf

STAGE 2 MODEL:
- Detects:
  - Apple___Apple_scab
  - Apple___Black_rot
  - Apple___Cedar_apple_rust
  - Apple___healthy

INPUT:
- 224x224 RGB images

PREPROCESSING:
- PIL image loading
- RGB conversion
- Resize
- NumPy conversion
- EfficientNet preprocessing

EXPLAINABILITY:
- Grad-CAM already implemented
- Multiple Grad-CAM files already exist

CURRENT PROJECT STRUCTURE:
- app.py
- unified_classifier.py
- stage1_classifier.py
- leaf_classifier.py
- config.py
- predict.py
- explainability/
- utils/
- models/
- test scripts
- Flask UI
- static outputs

========================================================
MAIN OBJECTIVE
========================================================

Your job is to help me COMPLETE THE PROJECT END-TO-END FOR FINAL SUBMISSION.

I need:
- evaluation outputs
- robustness experiments
- explainability validation
- deployment outputs
- figures
- plots
- metrics
- testing scripts
- organized outputs
- reproducible evaluation pipeline

DO NOT:
- rebuild the system
- redesign architecture unnecessarily
- suggest unrelated frameworks
- overcomplicate the workflow

========================================================
CRITICAL REQUIREMENTS
========================================================

The final implementation MUST include:

1. Confusion matrix
2. Precision / Recall / F1-score tables
3. Classification report
4. Training/validation accuracy curves
5. Training/validation loss curves
6. Grad-CAM visualizations
7. Quantitative Grad-CAM IoU evaluation
8. Robustness testing
9. Augmentation visualization figure
10. TFLite conversion
11. Latency benchmarking
12. Model size comparison
13. Quantized model evaluation
14. Organized outputs folder
15. Clean experiment pipeline
16. Reproducible scripts

========================================================
IMPORTANT RESEARCH CONTRIBUTION
========================================================

The MAIN novelty of the project is NOT classification alone.

The contribution is:
- explainability evaluation
- robustness benchmarking
- lightweight deployment
- real-world usability

Therefore:
- prioritize IoU-based Grad-CAM evaluation
- prioritize robustness testing
- prioritize deployment optimization

========================================================
EXPECTED PROJECT STRUCTURE
========================================================

I want the final project organized like this:

project/
│
├── data/
├── models/
├── outputs/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── augmentation_grid.png
│   ├── robustness/
│   ├── gradcam/
│   └── deployment/
│
├── annotations/
│
├── code/
│   ├── evaluate.py
│   ├── plot_training.py
│   ├── augmentation_figure.py
│   ├── gradcam_iou.py
│   ├── robustness_test.py
│   ├── convert_tflite.py
│   ├── benchmark_tflite.py
│   └── predict_single.py
│
└── paper/

========================================================
WHAT I NEED FROM YOU
========================================================

I want you to guide me STEP-BY-STEP from A-Z.

For EACH step:
- explain WHY it matters
- explain WHAT output it should generate
- explain HOW it connects to the research contribution
- explain HOW to run it in VS Code
- explain dependencies needed
- explain expected runtime
- explain expected outputs

Then generate COMPLETE PRODUCTION-READY CODE.

========================================================
IMPLEMENTATION TASKS
========================================================

TASK 1 — EVALUATION PIPELINE
Generate:
- evaluate.py

Requirements:
- load saved .keras model
- load test dataset
- compute:
  - accuracy
  - precision
  - recall
  - F1-score
- generate confusion matrix
- save confusion matrix heatmap
- save classification report
- support class names from JSON
- save outputs into outputs/

Also:
- explain how to run it
- explain expected outputs

========================================================

TASK 2 — TRAINING CURVES
Generate:
- plot_training.py

Requirements:
- load saved history.pkl if available
- plot:
  - train vs validation accuracy
  - train vs validation loss
- save figures
- if history does not exist:
  - explain fallback strategy

========================================================

TASK 3 — AUGMENTATION FIGURE
Generate:
- augmentation_figure.py

Requirements:
- show:
  - original
  - rotation
  - horizontal flip
  - brightness jitter
  - Gaussian noise
  - Gaussian blur
- create 2x3 matplotlib figure
- save as augmentation_grid.png

========================================================

TASK 4 — GRAD-CAM VISUALIZATION
Generate:
- gradcam_examples.py

Requirements:
- load saved model
- generate Grad-CAM
- overlay heatmap
- save outputs
- support batch generation
- save multiple example figures

========================================================

TASK 5 — QUANTITATIVE GRAD-CAM IoU EVALUATION
THIS IS THE MOST IMPORTANT PART.

Generate:
- gradcam_iou.py

Requirements:
- load annotation masks or bounding boxes
- convert Grad-CAM heatmap to binary mask
- compute IoU
- compute mean IoU
- save table/results
- generate visual comparison outputs

Explain:
- annotation workflow using LabelImg or CVAT
- how to export annotations
- how to convert annotations into masks

========================================================

TASK 6 — ROBUSTNESS TESTING
Generate:
- robustness_test.py

Requirements:
- apply:
  - Gaussian noise
  - blur
  - brightness shifts
- evaluate model at multiple severity levels
- plot:
  - severity vs accuracy
- save all figures
- support configurable distortions

========================================================

TASK 7 — TFLITE CONVERSION
Generate:
- convert_tflite.py

Requirements:
- quantize model
- export model_quantized.tflite
- compare:
  - original size
  - quantized size

========================================================

TASK 8 — LATENCY BENCHMARKING
Generate:
- benchmark_tflite.py

Requirements:
- benchmark inference latency
- compute average latency over 100 runs
- compare:
  - keras model
  - tflite model
- save benchmark results

========================================================

TASK 9 — OUTPUT ORGANIZATION
Generate:
- automatic folder creation logic
- clean saving system
- consistent naming conventions

========================================================

IMPORTANT IMPLEMENTATION RULES
========================================================

- Use TensorFlow/Keras
- Use Python 3.10+
- Use matplotlib
- Use seaborn
- Use sklearn.metrics
- Use OpenCV only where necessary
- Keep code modular
- Add comments
- Add logging
- Add exception handling
- Add path handling
- Add argparse if useful

========================================================
CRITICAL THINKING REQUIREMENT
========================================================

Do NOT blindly generate generic code.

You must:
- align scripts with my existing architecture
- align with EfficientNet preprocessing
- align with my current folder structure
- align with the two-stage system design
- align with the explainability modules already present

========================================================
FINAL OUTPUT FORMAT
========================================================

For EACH task:
1. Purpose
2. Why it matters
3. Expected outputs
4. Dependencies
5. Full code
6. How to run
7. Expected runtime
8. Common issues
9. Recommended improvements

========================================================
IMPORTANT
========================================================

- DO NOT skip any hidden implementation step.
- DO NOT simplify the project.
- DO NOT omit robustness or IoU evaluation.
- DO NOT give pseudo-code.
- GIVE COMPLETE RUNNABLE CODE.
- THINK like a senior ML engineer preparing a publication-grade submission.

Start with TASK 1 only.
Do not jump ahead.
Wait for confirmation after each task.

---

## 📊 Research Evaluation & Experimental Results

The following results were generated using the finalized evaluation pipeline on the **Apple Leaf Disease** dataset (Test set size: 2,694 images).

### 1. Stage 2: Disease Classification Performance
The model achieved high accuracy across all four disease categories.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Apple Scab** | 0.9535 | 0.8462 | 0.8966 | 702 |
| **Black Rot** | 0.9786 | 0.9956 | 0.9871 | 689 |
| **Cedar Apple Rust** | 1.0000 | 0.9328 | 0.9652 | 610 |
| **Healthy** | 0.8589 | 0.9928 | 0.9210 | 693 |
| **Macro Average** | **0.9477** | **0.9418** | **0.9425** | **2,694** |

**Overall Accuracy: 94.17%**

### 2. Robustness Benchmarking
Evaluation of model degradation under synthetic environmental corruptions.

| Distortion | Severity | Accuracy | Notes |
| :--- | :--- | :--- | :--- |
| **Gaussian Noise** | σ=75 | 0.5000 | Significant degradation at high noise |
| **Gaussian Blur** | k=9 | 0.9000 | Extremely stable under focus issues |
| **Brightness** | 0.4x | 0.8500 | Maintained performance in low light |
| **Brightness** | 1.6x | 0.9500 | High stability in overexposed conditions |

### 3. Explainability (Grad-CAM)
- **Qualitative**: Heatmaps consistently highlight lesion areas (scab spots, rust pustules).
- **Quantitative**: Scripts for IoU computation are ready for annotated ground-truth masks.

### 4. Deployment & Edge Optimization (TFLite)
The models were successfully compressed using multiple quantization strategies.

| Model Variant | Size (MB) | Compression | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Original (.keras)** | 16.33 MB | 1.00x | Server-side inference |
| **TFLite (float32)** | 15.33 MB | 1.07x | Standard mobile export |
| **TFLite (Dynamic)** | 4.36 MB | 3.74x | Balanced performance/size |
| **TFLite (Full Int8)**| 4.71 MB | 3.47x | TPU / Microcontroller |

---

## 🚀 Final Deliverables Checklist

- [x] **Evaluation Pipeline**: `code/evaluate.py`
- [x] **Visualization Suite**: `code/plot_training.py`, `code/augmentation_figure.py`
- [x] **Explainability Module**: `code/gradcam_examples.py` (Fixed for Keras 3)
- [x] **Robustness Suite**: `code/robustness_test.py`
- [x] **Deployment Suite**: `code/convert_tflite.py`, `code/benchmark_tflite.py`
- [x] **Single-Inference Script**: `code/predict_single.py` (Final production entry-point)

---

## 🛠 Progress & Context (Final Update)

- **Environment**: Upgraded to TensorFlow 2.17.0+ / Keras 3 to resolve model serialization issues.
- **Accuracy**: Confirmed 94.17% on Stage 2.
- **Deployment**: Full Int8 quantization achieved 3.47x size reduction with calibrated representative data.