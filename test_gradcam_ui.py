"""
Quick test script to verify Grad-CAM UI is working
"""

import requests
import json

# Test image path
test_image = "static/uploads/black_rot.jpg"

print("=" * 60)
print("TESTING GRAD-CAM UI INTEGRATION")
print("=" * 60)

# Open the image file
try:
    with open(test_image, 'rb') as f:
        files = {'leaf_image': f}
        
        print(f"\n📤 Uploading test image: {test_image}")
        print("-" * 60)
        
        # Make POST request to /predict endpoint
        response = requests.post('http://127.0.0.1:5001/predict', files=files)
        
        print(f"Response Status: {response.status_code}")
        
        # Parse JSON response
        data = response.json()
        
        print(f"\nResponse Data:")
        print(json.dumps(data, indent=2))
        
        print("\n" + "=" * 60)
        print("GRAD-CAM VISUALIZATION CHECK")
        print("=" * 60)
        
        if data.get('success'):
            if data.get('gradcam_image'):
                print(f"\n✅ SUCCESS: Grad-CAM image path returned!")
                print(f"   Image Path: {data['gradcam_image']}")
                print(f"   Stage: {data['stage']}")
                print(f"   Disease: {data.get('disease_display', 'N/A')}")
                print(f"   Confidence: {data.get('confidence', 0)*100:.1f}%")
                print(f"\n📸 The Grad-CAM visualization should load in the UI!")
            else:
                print(f"\n⚠️  WARNING: No gradcam_image in response")
                print(f"   Success: {data['success']}")
                print(f"   Stage: {data.get('stage', 'N/A')}")
        else:
            print(f"\n❌ FAILED: Prediction was not successful")
            print(f"   Error: {data.get('error', 'Unknown error')}")

except FileNotFoundError:
    print(f"\n❌ ERROR: Test image not found: {test_image}")
    print("\n📁 Available test images:")
    import os
    for img in os.listdir('static/uploads')[:5]:
        print(f"   - static/uploads/{img}")

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✨ TEST COMPLETE")
print("=" * 60)
