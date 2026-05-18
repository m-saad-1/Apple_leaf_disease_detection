import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

# Configuration
BASE_DIR = Path(r"D:\WEB DEVELOPMENT\Apple_leaf_detection")
ANNOTATIONS_DIR = BASE_DIR / "annotations"
XML_DIR = ANNOTATIONS_DIR / "labelimg_xml"
SELECTED_IMAGES_DIR = ANNOTATIONS_DIR / "selected_images"
CSV_PATH = ANNOTATIONS_DIR / "real_gradcam_annotations.csv"

# Class mappings
CLASS_MAPPING = {
    "Apple___Apple_scab": 0,
    "Apple___Black_rot": 1,
    "Apple___Cedar_apple_rust": 2
}

def xml_to_csv():
    xml_list = []
    xml_files = list(XML_DIR.glob("*.xml"))
    
    if not xml_files:
        print(f"No XML files found in {XML_DIR}")
        return

    print(f"Found {len(xml_files)} XML files.")

    total_annotations = 0
    skipped_annotations = 0
    missing_images = 0

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        filename_elem = root.find("filename")
        if filename_elem is None:
            print(f"Warning: No filename found in {xml_file.name}")
            continue
            
        filename = filename_elem.text
        
        # Searching for the file
        found_image_path = None
        found_class_name = None
        for cls in CLASS_MAPPING.keys():
            potential_path = SELECTED_IMAGES_DIR / cls / filename
            if potential_path.exists():
                found_image_path = potential_path
                found_class_name = cls
                break
        
        if not found_image_path:
            print(f"Warning: Image {filename} referenced in {xml_file.name} not found in {SELECTED_IMAGES_DIR}")
            missing_images += 1
            continue
            
        class_idx = CLASS_MAPPING[found_class_name]
        
        for member in root.findall("object"):
            label = member.find("name").text
            if label not in ["lesion", "labelw", "label"]:
                print(f"Warning: Skipping label '{label}' in {filename}. Expected 'lesion', 'labelw', or 'label'.")
                skipped_annotations += 1
                continue
                
            bndbox = member.find("bndbox")
            value = (
                str(found_image_path.absolute()), # image_path
                class_idx,                        # class_idx
                int(float(bndbox.find("xmin").text)),    # x1
                int(float(bndbox.find("ymin").text)),    # y1
                int(float(bndbox.find("xmax").text)),    # x2
                int(float(bndbox.find("ymax").text))     # y2
            )
            xml_list.append(value)
            total_annotations += 1

    if not xml_list:
        print("No valid annotations found. CSV not created.")
        return

    column_name = ["image_path", "class_idx", "x1", "y1", "x2", "y2"]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv(CSV_PATH, index=False)
    
    print("-" * 40)
    print("Conversion Summary:")
    print(f"Total annotations converted: {total_annotations}")
    print(f"Missing images: {missing_images}")
    print(f"Skipped annotations (not 'lesion'): {skipped_annotations}")
    print(f"Saved CSV to: {CSV_PATH}")

if __name__ == "__main__":
    xml_to_csv()
