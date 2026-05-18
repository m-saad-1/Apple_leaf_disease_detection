# Annotation Guide for Apple Leaf Disease Detection

This guide walks you through annotating the selected diseased leaf images for real Grad-CAM IoU evaluation.

## 1. Install Dependencies
First, ensure you have the required tools installed. Open your terminal and run:
```bash
pip install labelImg pandas opencv-python Pillow matplotlib
```

## 2. Launch LabelImg
Open the annotation tool using the following command:
```bash
labelImg
```
*(If the command doesn't work, try `python -m labelImg` instead.)*

## 3. Setup Folders in LabelImg
- **Open Dir**: Select the `annotations/selected_images/` folder (or open the subfolders individually).
- **Change Save Dir**: Set this to `annotations/labelimg_xml/`.
- Ensure the format is set to **PascalVOC/XML** (you can click the format button on the left panel to toggle it if it says YOLO).

## 4. Annotation Rules
Draw bounding boxes manually around the visible disease regions following these strict rules:

- **Label Name**: Use exactly `lesion` as the class name.
- **DO NOT** draw a box around:
  - The whole leaf
  - The whole image
  - The background
  - The stem
  - Healthy green areas
  - Shadows
- **DO** draw a box around:
  - Scab spots
  - Black rot patches
  - Cedar rust spots
  - Main visible disease clusters

*(You can draw multiple boxes per image if there are multiple separated lesions).*

## 5. Save Annotations
Make sure to save the annotation for every image before moving to the next one.

## 6. Process Annotations
After finishing all annotations, convert them to a CSV file for the notebook:
```bash
python scripts/xml_to_csv.py
```
This script reads all XML files and creates `annotations/real_gradcam_annotations.csv`.

## 7. Verify Annotations
Run the checking script to validate the CSV data and generate preview images:
```bash
python scripts/check_annotations.py
```
Check `annotations/previews/` to verify that your bounding boxes look correct on the images.

## 8. Use in Notebook
Finally, use the generated CSV in your Jupyter notebook for real Grad-CAM IoU evaluation. Update the path in the notebook:
```python
ANNOTATION_CSV = "annotations/real_gradcam_annotations.csv"
# Or on Windows, if you need an absolute path:
# ANNOTATION_CSV = r"D:\WEB DEVELOPMENT\Apple_leaf_detection\annotations\real_gradcam_annotations.csv"
```
