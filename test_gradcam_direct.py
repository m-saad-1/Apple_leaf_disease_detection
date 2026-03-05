"""
Direct test of Grad-CAM generation to see the actual error
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from explainability import generate_disease_focused_gradcam
from leaf_classifier import stage2_model

print("=" * 60)
print("TESTING DISEASE-FOCUSED GRAD-CAM DIRECTLY")
print("=" * 60)

test_image = "static/uploads/black_rot.jpg"
print(f"\nTesting with image: {test_image}")
print(f"   Image exists: {os.path.exists(test_image)}")

print("\nCalling generate_disease_focused_gradcam...")
print("-" * 60)

try:
    result = generate_disease_focused_gradcam(
        image_path=test_image,
        model=stage2_model,
        output_dir="static/gradcam",
        mode="blend"
    )
    
    print(f"\nResult returned: {result}")
    
    if result:
        print(f"   Path exists: {os.path.exists(result)}")
        if os.path.exists(result):
            size = os.path.getsize(result)
            print(f"   File size: {size} bytes")
    
except Exception as e:
    print(f"\nException occurred:")
    print(f"   Type: {type(e).__name__}")
    print(f"   Message: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
