# Apple Leaf Disease Detection: Research Paper Details

Last reviewed: 2026-05-30

## 1. Study Scope and Goal

This repository implements an explainable deep-learning system for apple leaf disease analysis using a two-stage inference pipeline and Grad-CAM-based visual explanations.

Primary goal:
- Detect whether an uploaded image is an apple leaf and triage healthy vs diseased (Stage 1).
- Classify disease type for diseased leaves (Stage 2).
- Provide visual evidence (Grad-CAM) to support interpretation.

Intended paper focus:
- Classification performance.
- Explainability quality (localization against lesion annotations).
- Robustness under image distortions.
- Deployment feasibility with TFLite compression.

## 2. Project Structure (Reviewed)

Core inference stack:
- `app.py`: Flask server and `/predict` API.
- `unified_classifier.py`: end-to-end two-stage orchestration.
- `stage1_classifier.py`: Stage 1 prediction wrapper.
- `leaf_classifier.py`: Stage 2 prediction wrapper.
- `utils/image_processing.py`: shared preprocessing.
- `explainability/gradcam_disease_focused.py`: disease-focused Grad-CAM pipeline.

Evaluation and analysis:
- `code/evaluate.py`: Stage 2 test-set evaluation.
- `code/robustness_test.py`: perturbation robustness checks.
- `code/evaluate_tflite.py`: TFLite accuracy evaluation.
- `outputs/`: generated figures, reports, and CSV metrics.
- `annotations/`: lesion annotations and conversion outputs.

## 3. Dataset and Splits

On-disk split counts (verified from `dataset/`):

| Split | Apple scab | Black rot | Cedar apple rust | Healthy | Total |
|---|---:|---:|---:|---:|---:|
| Train | 2294 | 2257 | 2007 | 2286 | 8844 |
| Validation | 702 | 690 | 614 | 698 | 2704 |
| Test | 702 | 689 | 610 | 693 | 2694 |
| **Grand total** | 3698 | 3636 | 3231 | 3677 | **14242** |

Annotation dataset for XAI evaluation (`annotations/real_gradcam_annotations.csv`):
- 168 unique annotated images.
- 542 lesion bounding boxes.
- Class-wise boxes:
  - Class 0 (Apple scab): 155
  - Class 1 (Black rot): 153
  - Class 2 (Cedar apple rust): 234

## 4. Methodology

### 4.1 Two-stage decision flow

1. Stage 1 predicts one of:
- `Apple_Diseased`
- `Apple_Healthy`
- `Not_Apple_Leaf`

2. Decision logic:
- If `Not_Apple_Leaf`: reject image.
- If `Apple_Healthy`: return healthy result.
- If `Apple_Diseased`: run Stage 2 disease classifier.

3. Stage 2 predicts one of:
- `Apple___Apple_scab`
- `Apple___Black_rot`
- `Apple___Cedar_apple_rust`
- `Apple___healthy`

### 4.2 Preprocessing

Shared preprocessing (`utils/image_processing.py`):
- RGB conversion.
- Center crop to square.
- Resize to 224x224.
- EfficientNet preprocessing.

### 4.3 Thresholding policy (from `config.py`)

- Stage 1:
  - Confidence threshold = 0.70
  - Margin threshold = 0.15
- Stage 2:
  - Confidence threshold = 0.75
  - Margin threshold = 0.20

A prediction is treated as reliable only if both confidence and top-2 margin pass thresholds.

### 4.4 Explainability approach

`gradcam_disease_focused.py` adds post-processing beyond plain Grad-CAM:
- Edge suppression via Sobel-based mask.
- Dark-region suppression.
- Morphological filtering.
- Contrast enhancement (CLAHE).

This is designed to reduce edge-focused saliency and emphasize lesion-like regions.

## 5. Quantitative Results (Available Artifacts)

### 5.1 Stage 2 classification performance

From `outputs/classification_report_stage2.txt`:
- Overall Accuracy: **0.9417** on 2694 test images.

Class-wise metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Apple___Apple_scab | 0.9535 | 0.8462 | 0.8966 | 702 |
| Apple___Black_rot | 0.9786 | 0.9956 | 0.9871 | 689 |
| Apple___Cedar_apple_rust | 1.0000 | 0.9328 | 0.9652 | 610 |
| Apple___healthy | 0.8589 | 0.9928 | 0.9210 | 693 |
| **Macro avg** | 0.9477 | 0.9418 | 0.9425 | 2694 |

Additional training log artifact (`temp_accuracy.txt`) reports:
- Test Accuracy: 0.9480 (94.80%)
- This suggests multiple runs/checkpoints were evaluated.

### 5.2 Robustness under distortions

From `outputs/robustness/robustness_results_stage_2.csv`:
- Gaussian Noise: accuracy drops from 0.95 (severity 0) to 0.50 (severity 75).
- Gaussian Blur: mostly robust up to high severity (0.90 at severity 9).
- Brightness Shift: robust across moderate shifts; 0.85 at severity 0.4.

From `outputs/occlusion_robustness_results.csv`:
- Occlusion 10%: 0.9443
- Occlusion 20%: 0.9347
- Occlusion 30%: 0.9065
- Occlusion 40%: 0.8545

Robustness AUC summary (`outputs/robustness_auc_summary.csv`):
- Occlusion AUC: **0.9135**

### 5.3 Explainability localization results

From `outputs/real_gradcam_iou_results.csv` (111 lesion boxes evaluated):
- Mean Grad-CAM IoU: **0.0308**
- Mean lesion coverage: **0.3276**

Per-class summary:

| Class | Boxes | Mean IoU | Mean Coverage | Non-zero IoU boxes |
|---|---:|---:|---:|---:|
| Apple scab | 33 | 0.0594 | 0.3876 | 20 |
| Black rot | 27 | 0.0312 | 0.6107 | 20 |
| Cedar apple rust | 51 | 0.0121 | 0.1389 | 18 |

Interpretation for paper discussion:
- Localization overlap is currently low (IoU), despite moderate lesion coverage.
- This supports a nuanced conclusion: class prediction can remain strong while explanation localization is less precise.

### 5.4 Deployment and compression

From `outputs/deployment/conversion_summary.txt` and `model_size_comparison.csv`:
- Approximate Keras size per model: 16.32-16.33 MB
- TFLite float32: 15.33 MB
- TFLite dynamic quantized: 4.36 MB
- Compression ratio (quantized vs Keras): ~3.74x (about 73.3% size reduction)

From `temp_accuracy.txt` deployment summary:
- Keras latency: 170.19 ms (reported in log summary table artifact)
- TFLite FP32 latency: 27.18 ms
- TFLite quantized latency: 52.88 ms
- Quantized TFLite quick-check accuracy (10 batches): 85.94%

## 6. Reproducibility Commands

Environment:
```bash
pip install -r requirements.txt
```

Run server:
```bash
python app.py
```

Stage 2 evaluation:
```bash
python code/evaluate.py --model models/stage_2.keras --classes models/class_names.json --dataset dataset/test --output outputs
```

Robustness test:
```bash
python code/robustness_test.py --model models/stage_2.keras --dataset dataset/test --output outputs/robustness
```

TFLite evaluation:
```bash
python code/evaluate_tflite.py --tflite_path outputs/deployment/stage2_model_quantized.tflite --data_dir dataset/test
```

## 7. Complete Project Review (Important Findings)

### 7.1 Current reproducibility blockers

1. Model filename mismatch (critical)
- `config.py` expects:
  - `models/stage_1.keras`
  - `models/stage_2.keras`
- Actual files present:
  - `models/stage_1.keras`
  - `models/stage_2.keras`
- Running `python config.py` currently reports missing models.

2. Test script encoding failure on Windows console
- `python test_system.py` fails with `UnicodeEncodeError` due checkmark characters under cp1252 console encoding.
- This prevents complete automated validation unless UTF-8 output is enforced.

3. Inconsistent historical naming across files
- Some scripts/docs refer to `stage_2.keras`, others to `stage_2.keras`, while current directory has `stage_2.keras`.
- Same issue appears for Stage 1 naming (`stage_1.keras` vs `stage_1.keras`).

### 7.2 Methodological limitations to declare in paper

- Stage 1 quantitative metrics are not packaged as a formal report artifact in `outputs/`.
- Training pipeline scripts are not fully present; reproducibility currently relies on saved artifacts/logs.
- Explainability localization scores (IoU) are low, so claims should avoid overstating lesion-level alignment.
- Some robustness evaluations appear to run on sampled subsets (script-level speed limits), which should be reported transparently.

## 8. Recommended Pre-submission Cleanup

1. Standardize model filenames and paths across `config.py`, scripts, tests, and docs.
2. Add a single reproducibility script that runs all evaluations and exports a consolidated metrics table.
3. Add Stage 1 evaluation report (accuracy, per-class precision/recall/F1, confusion matrix).
4. Fix console encoding in tests (or remove Unicode symbols) for cross-platform execution.
5. Add an experiment manifest (`seed`, hardware, package versions, dataset checksum) for paper appendix.

## 9. Suggested Paper Positioning (Based on Current Evidence)

Strong claims supported:
- High Stage 2 classification performance on test split.
- Significant model size reduction via quantization.
- Robustness degrades gracefully for occlusion and brightness, but noise is a major failure mode.

Careful claims advised:
- Explainability quality should be discussed as partial/limited due low IoU.
- End-to-end reproducibility currently requires path harmonization and test-script cleanup.

## 10. Artifact Pointers

Main result artifacts:
- `outputs/classification_report_stage2.txt`
- `outputs/metrics_stage2.csv`
- `outputs/confusion_matrix_stage2.png`
- `outputs/robustness/robustness_results_stage_2.csv`
- `outputs/occlusion_robustness_results.csv`
- `outputs/robustness_auc_summary.csv`
- `outputs/real_gradcam_iou_results.csv`
- `outputs/deployment/conversion_summary.txt`
- `outputs/deployment/model_size_comparison.csv`
- `temp_accuracy.txt`

Methodology figures:
- `methodology_figures/Fig1_System_Pipeline.png`
- `methodology_figures/Fig2_EfficientNetB0_Architecture.png`
- `methodology_figures/Fig3_XAI_Methods_Technical.png`
- `methodology_figures/Fig4_Annotation_Evaluation.png`
- `methodology_figures/Fig5_Robustness_Protocol.png`
- `methodology_figures/Fig6_Methodology_Summary.png`

