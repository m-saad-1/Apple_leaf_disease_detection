You are a senior Python ML project engineer. I am working on an Explainable Apple Leaf Disease Detection research project. I need you to prepare my project for real Grad-CAM IoU annotation.

Important context:
- I already have a trained EfficientNetB0 apple leaf disease model.
- I already have a final Jupyter notebook.
- I now need to manually annotate 30+ diseased apple leaf images using LabelImg.
- The goal is to create real bounding-box annotations around visible disease regions for Grad-CAM IoU evaluation.
- Do NOT train any model.
- Do NOT modify the model architecture.
- Do NOT change the notebook unless required for annotation path compatibility.
- Your job is only to prepare the annotation workflow, folders, files, scripts, and instructions.

Project classes:
1. Apple___Apple_scab
2. Apple___Black_rot
3. Apple___Cedar_apple_rust
4. Apple___healthy

For annotation, use ONLY diseased classes:
- Apple___Apple_scab
- Apple___Black_rot
- Apple___Cedar_apple_rust

Do NOT use Apple___healthy for lesion IoU because healthy leaves do not contain disease regions.

====================================================
TASK 1 — CREATE ANNOTATION FOLDER STRUCTURE
====================================================

Create this folder structure in the project root:

annotations/
├── selected_images/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   └── Apple___Cedar_apple_rust/
├── labelimg_xml/
├── previews/
└── real_gradcam_annotations.csv

scripts/
├── prepare_annotation_images.py
├── xml_to_csv.py
└── check_annotations.py

docs/
└── ANNOTATION_GUIDE.md

If folders already exist, do not delete existing files. Create missing folders safely.

====================================================
TASK 2 — CREATE IMAGE SELECTION SCRIPT
====================================================

Create scripts/prepare_annotation_images.py.

Requirements:
- It should copy 30+ diseased test images from the existing dataset into annotations/selected_images/.
- Select 10 images from each disease class:
  - 10 from Apple___Apple_scab
  - 10 from Apple___Black_rot
  - 10 from Apple___Cedar_apple_rust
- It should NOT select Apple___healthy images.
- It should support configurable dataset path at the top of the file.
- It should search inside the test folder first.
- If test folder is not found, allow user to configure source folders manually.
- It should preserve class subfolders.
- It should rename copied images clearly, for example:
  - scab_001.jpg
  - black_rot_001.jpg
  - cedar_rust_001.jpg
- It should avoid overwriting existing selected images unless explicitly enabled.
- It should print a summary:
  - number of images copied per class
  - destination folder
  - warning if any class has fewer than 10 images
- Use pathlib and shutil.
- Add clear comments and error handling.

Expected output:
annotations/selected_images/
├── Apple___Apple_scab/
│   ├── scab_001.jpg
│   └── ...
├── Apple___Black_rot/
│   ├── black_rot_001.jpg
│   └── ...
└── Apple___Cedar_apple_rust/
    ├── cedar_rust_001.jpg
    └── ...

====================================================
TASK 3 — CREATE XML TO CSV CONVERSION SCRIPT
====================================================

Create scripts/xml_to_csv.py.

Requirements:
- Read PascalVOC XML files exported by LabelImg from annotations/labelimg_xml/.
- Convert them into:
  annotations/real_gradcam_annotations.csv
- CSV columns must be exactly:
  image_path,class_idx,x1,y1,x2,y2
- class_idx mapping:
  Apple___Apple_scab = 0
  Apple___Black_rot = 1
  Apple___Cedar_apple_rust = 2
- The script should locate the original image inside annotations/selected_images/class_name/.
- Only accept object label:
  lesion
- If any label is not "lesion", print a warning and skip it.
- If XML references an image that cannot be found, print a warning.
- Support multiple boxes per image if multiple lesion boxes exist.
- Save absolute image paths if possible, because the notebook may need to load them directly.
- Print total annotations converted.
- Print total images found.
- Print any missing or skipped files.
- Use pandas and xml.etree.ElementTree.

Expected CSV example:
image_path,class_idx,x1,y1,x2,y2
D:/project/annotations/selected_images/Apple___Apple_scab/scab_001.jpg,0,45,60,180,200
D:/project/annotations/selected_images/Apple___Black_rot/black_rot_001.jpg,1,50,70,190,210
D:/project/annotations/selected_images/Apple___Cedar_apple_rust/cedar_rust_001.jpg,2,40,55,175,185

====================================================
TASK 4 — CREATE ANNOTATION CHECKING SCRIPT
====================================================

Create scripts/check_annotations.py.

Requirements:
- Load annotations/real_gradcam_annotations.csv.
- Check:
  - CSV exists
  - required columns exist
  - image paths exist
  - class_idx values are only 0, 1, 2
  - bounding boxes are valid:
    x2 > x1
    y2 > y1
    coordinates are non-negative
- Print class-wise annotation counts.
- Print total annotations.
- Print invalid rows if any.
- Create preview images in annotations/previews/.
- Preview images should show bounding boxes drawn on top of the selected images.
- Use OpenCV or PIL.
- Do not modify original images.

====================================================
TASK 5 — INSTALLATION / DEPENDENCIES
====================================================

Update requirements.txt or create annotation_requirements.txt with:

labelImg
pandas
opencv-python
Pillow
matplotlib

Do not remove existing project dependencies.

Also add comments or instructions for installation:

pip install labelImg pandas opencv-python Pillow matplotlib

If LabelImg does not launch using labelImg, mention alternative:

python -m labelImg

====================================================
TASK 6 — CREATE ANNOTATION GUIDE
====================================================

Create docs/ANNOTATION_GUIDE.md.

The guide must explain in simple steps:

1. Install LabelImg:
   pip install labelImg

2. Open LabelImg:
   labelImg
   or:
   python -m labelImg

3. Open image folder:
   annotations/selected_images/

4. Change save directory:
   annotations/labelimg_xml/

5. Use PascalVOC/XML format.

6. Draw bounding boxes manually around visible disease regions.

7. Use only this label:
   lesion

8. Do NOT draw box around:
   - whole leaf
   - whole image
   - background
   - stem
   - healthy green area
   - shadow

9. Draw box around:
   - scab spots
   - black rot patches
   - cedar rust spots
   - main visible disease cluster

10. Save every image annotation.

11. After finishing annotation, run:
    python scripts/xml_to_csv.py

12. Then run:
    python scripts/check_annotations.py

13. Then use:
    annotations/real_gradcam_annotations.csv
    inside the notebook for real Grad-CAM IoU evaluation.

====================================================
TASK 7 — ADD NOTEBOOK PATH COMPATIBILITY NOTE
====================================================

Do not rewrite the notebook, but add a small helper note or code snippet in docs/ANNOTATION_GUIDE.md explaining that the notebook should use:

ANNOTATION_CSV = "annotations/real_gradcam_annotations.csv"

or, on Windows:

ANNOTATION_CSV = r"FULL_PATH_TO_PROJECT/annotations/real_gradcam_annotations.csv"

====================================================
TASK 8 — DO NOT OVERDO
====================================================

Do not add unnecessary tools.
Do not create a new model.
Do not retrain anything.
Do not change Flask application logic.
Do not delete existing files.
Do not create fake annotations.
Do not auto-generate bounding boxes.
I will manually draw the boxes myself.

====================================================
FINAL EXPECTED RESULT
====================================================

After your work, I should be able to run:

1. python scripts/prepare_annotation_images.py

This should copy 30 diseased images into:
annotations/selected_images/

2. labelImg

I will manually draw boxes and save XML files into:
annotations/labelimg_xml/

3. python scripts/xml_to_csv.py

This should create:
annotations/real_gradcam_annotations.csv

4. python scripts/check_annotations.py

This should verify the CSV and create preview images.

Then I will use the CSV inside my final notebook for Grad-CAM IoU evaluation.

Please implement this carefully, cleanly, and safely.