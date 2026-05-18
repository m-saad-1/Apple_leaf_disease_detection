import os
import pandas as pd
import cv2
from pathlib import Path

# Configuration
BASE_DIR = Path(r"D:\WEB DEVELOPMENT\Apple_leaf_detection")
ANNOTATIONS_DIR = BASE_DIR / "annotations"
CSV_PATH = ANNOTATIONS_DIR / "real_gradcam_annotations.csv"
PREVIEWS_DIR = ANNOTATIONS_DIR / "previews"

def check_annotations():
    # 1. Check if CSV exists
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # 2. Check required columns
    required_cols = ["image_path", "class_idx", "x1", "y1", "x2", "y2"]
    if list(df.columns) != required_cols:
        print(f"Error: CSV columns do not match. Expected: {required_cols}. Found: {list(df.columns)}")
        return

    print("CSV format is valid.")
    
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    
    invalid_rows = []
    class_counts = {0: 0, 1: 0, 2: 0}
    
    # Group by image path to draw all boxes on one image
    grouped = df.groupby("image_path")
    
    for img_path_str, group in grouped:
        img_path = Path(img_path_str)
        
        if not img_path.exists():
            print(f"Warning: Image not found {img_path}")
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        for idx, row in group.iterrows():
            c_idx = row["class_idx"]
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            
            # Validation
            if c_idx not in [0, 1, 2]:
                invalid_rows.append((idx, "Invalid class_idx", row))
                continue
                
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                invalid_rows.append((idx, "Invalid bounding box coordinates", row))
                continue
                
            class_counts[c_idx] += 1
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "lesion", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # Save preview after drawing all boxes for this image
        preview_name = f"preview_{img_path.name}"
        cv2.imwrite(str(PREVIEWS_DIR / preview_name), img)
            
    print("-" * 40)
    print("Validation Summary:")
    print(f"Total annotations: {len(df)}")
    print("Class-wise annotation counts:")
    print(f"  Class 0 (Apple Scab): {class_counts[0]}")
    print(f"  Class 1 (Black Rot): {class_counts[1]}")
    print(f"  Class 2 (Cedar Rust): {class_counts[2]}")
    
    if invalid_rows:
        print("\nInvalid rows found:")
        for r in invalid_rows:
            print(f"Row {r[0]}: {r[1]} - Data: {r[2].to_dict()}")
    else:
        print("\nAll annotations are valid.")
        
    print(f"Previews saved to {PREVIEWS_DIR}")

if __name__ == "__main__":
    check_annotations()
