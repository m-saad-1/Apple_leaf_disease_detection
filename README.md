# 🍎 Apple Leaf Disease Detection System

A deep learning-based Flask web application for detecting diseases in apple leaves using a two-stage classification pipeline with explainability through Grad-CAM visualization.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 🌟 Features

- **Two-Stage Classification Pipeline**
  - Stage 1: Apple Leaf Detection (Apple/Not Apple)
  - Stage 2: Disease Classification (Healthy/Diseased with specific disease identification)

- **Web Interface**: Flask-based user-friendly web application for image uploads and predictions

- **Explainability**: Grad-CAM visualization to understand model predictions

- **Real-time Processing**: Fast inference on CPU and GPU

- **Configurable Thresholds**: Easily adjust confidence and margin thresholds for both stages

- **Image Preprocessing**: Robust image handling with center cropping and resizing

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input (Image)                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────▼───────────────┐
        │   Stage 1: Apple Detection     │
        │  (EfficientNet-based Classifier)
        └───────────────┬───────────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
    ┌────▼────┐  ┌─────▼──┐  ┌────────▼─────┐
    │  Health │  │Diseased│  │ Not an Apple │
    │   Leaf  │  │  Leaf  │  │     Leaf     │
    └────┬────┘  └────┬───┘  └────────┬─────┘
         │            │              │
    ┌────▼────┐   ┌───▼──────┐   ┌───▼──────┐
    │ Return  │   │  Stage 2 │   │  Reject  │
    │Healthy  │   │ Disease  │   │  Request │
    └─────────┘   │Classifier│   └──────────┘
                  └────┬─────┘
                       │
            ┌──────────┴──────────┐
            │                     │
        ┌───▼────┐         ┌──────▼────┐
        │Disease │         │Grad-CAM   │
        │Type    │         │Heatmap    │
        │(+ conf)│         │(Optional) │
        └────────┘         └───────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- 2GB+ free disk space (for models)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/apple-leaf-detection.git
   cd apple-leaf-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify configuration**
   ```bash
   python config.py
   ```

## 🚀 Quick Start

### Web Interface

1. **Start the Flask application**
   ```bash
   python app.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:5001`
   - Upload an apple leaf image (PNG, JPG, JPEG)
   - View prediction results with confidence scores and optional Grad-CAM visualization

3. **Stop the application**
   - Press `Ctrl+C` in the terminal

### Command Line Interface

Use the command-line prediction script:

```bash
python predict.py path/to/your/image.jpg
```

Output:
```
Prediction Result:
stage1_prediction: Apple_Diseased
stage1_confidence: 0.94
disease_name: Powdery Mildew
disease_confidence: 0.89
gradcam_available: True
```

## ⚙️ Configuration

Edit [config.py](config.py) to customize:

### Flask Settings
```python
FLASK_HOST = '0.0.0.0'      # Server host
FLASK_PORT = 5001            # Server port
FLASK_DEBUG = True           # Debug mode
```

### Model Paths
```python
STAGE1_MODEL_PATH = "models/stage1_model.keras"
STAGE2_MODEL_PATH = "models/leaf_model2.keras"
```

### Classification Thresholds

**Stage 1 Configuration:**
```python
STAGE1_CONFIG = {
    "confidence_threshold": 0.70,
    "margin_threshold": 0.15,
    "target_size": (224, 224),
    "center_crop": True,
}
```

**Stage 2 Configuration:**
```python
STAGE2_CONFIG = {
    "confidence_threshold": 0.65,
    "margin_threshold": 0.10,
    "target_size": (256, 256),
}
```

Adjust these thresholds based on your use case:
- **Higher confidence_threshold**: More conservative predictions
- **Higher margin_threshold**: Requires larger gap between top predictions

## 📁 Project Structure

```
apple-leaf-detection/
├── app.py                          # Flask application entry point
├── config.py                       # Configuration module
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── Core Modules
├── unified_classifier.py           # Two-stage prediction pipeline
├── stage1_classifier.py            # Stage 1: Apple leaf detection
├── leaf_classifier.py              # Stage 2: Disease classification
├── predict.py                      # CLI prediction script
│
├── Explainability
├── explainability/
│   ├── gradcam.py                  # Base Grad-CAM implementation
│   ├── gradcam_simple.py           # Simplified Grad-CAM
│   ├── gradcam_enhanced.py         # Enhanced Grad-CAM
│   └── gradcam_disease_focused.py  # Disease-focused Grad-CAM
│
├── Utilities
├── utils/
│   └── image_processing.py         # Image preprocessing utilities
│
├── Models & Data
├── models/
│   ├── stage1_model.keras          # Stage 1 model
│   ├── stage1_class_names.json     # Stage 1 class labels
│   ├── leaf_model2.keras           # Stage 2 model
│   └── class_names.json            # Stage 2 class labels
│
├── Web Interface
├── templates/
│   ├── index.html                  # Main upload page
│   └── 404.html                    # Error page
├── static/
│   ├── css/
│   │   └── style.css               # Styling
│   ├── gradcam/                    # Grad-CAM visualizations
│   └── uploads/                    # Uploaded images
│
├── Testing
├── test_system.py                  # System integration tests
├── test_gradcam_direct.py          # Grad-CAM direct testing
└── test_gradcam_ui.py              # Grad-CAM UI testing
```

## 🤖 Models

### Stage 1: Apple Leaf Detection
- **Architecture**: EfficientNet-based classifier
- **Input Size**: 224×224 pixels
- **Classes**:
  - Apple_Healthy: Healthy apple leaf
  - Apple_Diseased: Diseased apple leaf
  - Not_Apple_Leaf: Non-apple leaf or background

### Stage 2: Disease Classification
- **Architecture**: Custom CNN
- **Input Size**: 256×256 pixels
- **Output**: Specific disease classification with confidence score

## 📊 Model Performance

| Stage | Metric | Value |
|-------|--------|-------|
| Stage 1 | Accuracy | ~94% |
| Stage 1 | Precision | ~92% |
| Stage 2 | Accuracy | ~89% |
| Stage 2 | Precision | ~88% |

*Note: Performance metrics based on validation dataset*

## 🔍 Usage Examples

### Python Integration

```python
from unified_classifier import predict_leaf_disease

# Single prediction
result = predict_leaf_disease('path/to/leaf.jpg')

print(f"Stage 1 Prediction: {result['stage1_prediction']}")
print(f"Stage 1 Confidence: {result['stage1_confidence']:.4f}")

if result['stage1_prediction'] == 'Apple_Diseased':
    print(f"Disease: {result['disease_name']}")
    print(f"Disease Confidence: {result['disease_confidence']:.4f}")
```

### API Usage

**Endpoint**: `POST /predict`

**Request**:
```bash
curl -X POST -F "file=@path/to/leaf.jpg" http://localhost:5001/predict
```

**Response**:
```json
{
  "stage1_prediction": "Apple_Diseased",
  "stage1_confidence": 0.94,
  "disease_name": "Powdery Mildew",
  "disease_confidence": 0.89,
  "gradcam_url": "/static/gradcam/cam_1234567890.png",
  "status": "success"
}
```

## 🧪 Testing

Run the test suite:

```bash
# System integration test
python test_system.py

# Direct Grad-CAM test
python test_gradcam_direct.py

# UI Grad-CAM test
python test_gradcam_ui.py
```

## 📝 Logging

All application logs are stored in the `logs/` directory. Check logs for debugging:

```bash
tail -f logs/app.log
```

## 🛠️ Troubleshooting

### Model Files Not Found
- Ensure models are in `models/` directory
- Check [config.py](config.py) for correct model paths
- Verify file permissions

### Out of Memory
- Reduce `MAX_CONTENT_LENGTH` in app.py
- Process smaller images
- Use GPU if available

### Slow Predictions
- Enable GPU acceleration (TensorFlow/CUDA)
- Reduce image resolution
- Batch process multiple images

### Prediction Accuracy Issues
- Review confidence thresholds in config.py
- Validate input image quality
- Check stage1/stage2 classification separately

## 🔧 Development

### Adding New Disease Classes

1. Retrain Stage 2 model with new data
2. Update class names in `models/class_names.json`
3. Adjust confidence thresholds in `config.py`
4. Test with new disease samples

### Custom Image Preprocessing

Edit [utils/image_processing.py](utils/image_processing.py) to modify:
- Image resizing algorithms
- Normalization methods
- Augmentation techniques

### Extending Grad-CAM

Customize explainability in `explainability/` directory:
- Modify heatmap colormap
- Adjust overlay transparency
- Implement custom visualization

## 📈 Performance Optimization

- **GPU Acceleration**: Install CUDA and cuDNN for faster inference
- **Model Quantization**: Convert models to int8 for reduced memory footprint
- **Batch Processing**: Process multiple images simultaneously
- **Caching**: Cache predictions for identical images

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Write unit tests for new features
- Update README for significant changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💼 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- Flask for web framework
- EfficientNet for transfer learning
- Grad-CAM for model explainability
- OpenCV for image processing

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02055)

## ⚠️ Disclaimer

This system is designed for educational and research purposes. For agricultural production decisions, consult with agricultural experts and agronomists. The model predictions should be validated by domain experts before taking any action.

## 📞 Support

For issues, questions, or suggestions:
- Open an Issue on GitHub
- Email: your.email@example.com
- Check existing documentation and FAQs

---

**Last Updated**: March 2026  
**Version**: 1.0.0
