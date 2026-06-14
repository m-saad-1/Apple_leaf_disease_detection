import os
import shutil
from pathlib import Path

# Configurable dataset path - user can change this
DATASET_PATH = Path(r"D:\WEB DEVELOPMENT\Apple_leaf_detection\dataset")
TEST_FOLDER = DATASET_PATH / "test" # typically dataset/test

# Destination folder
BASE_DIR = Path(r"D:\WEB DEVELOPMENT\Apple_leaf_detection")
ANNOTATIONS_DIR = BASE_DIR / "annotations"
SELECTED_IMAGES_DIR = ANNOTATIONS_DIR / "selected_images"

DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust"
]

IMAGES_PER_CLASS = 50

def prepare_images():
    print(f"Checking dataset path: {TEST_FOLDER} (or {DATASET_PATH})")
    
    # Verify dataset exists
    if TEST_FOLDER.exists():
        search_dir = TEST_FOLDER
    elif DATASET_PATH.exists():
        search_dir = DATASET_PATH
        print("Test folder not found, using root dataset path.")
    else:
        print(f"Error: Dataset path not found: {DATASET_PATH}")
        print("Please configure DATASET_PATH manually in the script.")
        return

    # Create selected images folders
    for cls in DISEASE_CLASSES:
        (SELECTED_IMAGES_DIR / cls).mkdir(parents=True, exist_ok=True)
    
    total_copied = 0

    for cls in DISEASE_CLASSES:
        class_dir = search_dir / cls
        if not class_dir.exists():
            print(f"Warning: Class folder not found: {class_dir}")
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        
        if len(images) < IMAGES_PER_CLASS:
            print(f"Warning: Class {cls} has only {len(images)} images (less than {IMAGES_PER_CLASS} required).")
        
        # Select first IMAGES_PER_CLASS
        selected = images[:IMAGES_PER_CLASS]
        copied_for_class = 0
        
        for idx, img_path in enumerate(selected, start=1):
            # Formulate simple name, e.g., scab_001.jpg
            if "scab" in cls.lower():
                prefix = "scab"
            elif "black_rot" in cls.lower():
                prefix = "black_rot"
            elif "cedar" in cls.lower() or "rust" in cls.lower():
                prefix = "cedar_rust"
            else:
                prefix = "img"
                
            new_name = f"{prefix}_{idx:03d}{img_path.suffix}"
            dest_path = SELECTED_IMAGES_DIR / cls / new_name
            
            if not dest_path.exists():
                shutil.copy(img_path, dest_path)
                copied_for_class += 1
            else:
                print(f"File {dest_path.name} already exists. Skipping to avoid overwrite.")
        
        print(f"Class {cls}: Copied {copied_for_class} images.")
        total_copied += copied_for_class

    print("-" * 40)
    print(f"Summary:")
    print(f"Total images copied: {total_copied}")
    print(f"Destination folder: {SELECTED_IMAGES_DIR}")

if __name__ == "__main__":
    prepare_images()
