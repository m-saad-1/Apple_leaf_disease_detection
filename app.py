"""
Flask Application: Apple Leaf Disease Detection System
Implements a two-stage classification pipeline for detecting apple leaf diseases.

Routes:
  GET  /              - Home page with upload interface
  POST /predict       - Submit image for disease classification
  ALL  /404           - Custom 404 error page
"""

from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from unified_classifier import predict_leaf_disease
from config import (
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    SECRET_KEY,
    validate_config
)

app = Flask(__name__)

# Validate configuration on startup
errors = validate_config()
if errors:
    print("WARNING: Configuration validation errors:")
    for error in errors:
        print(f"  - {error}")

# Configuration
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['SECRET_KEY'] = SECRET_KEY


def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    """Render the home page with upload interface."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and prediction.
    
    Expects: POST with 'leaf_image' file in form data
    Returns: JSON with prediction results from two-stage pipeline
    """
    # Validate file presence
    if 'leaf_image' not in request.files:
        return jsonify({
            "success": False,
            "error": "No file part in request",
            "stage": 0
        }), 400
    
    file = request.files['leaf_image']
    
    # Validate filename
    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "No file selected",
            "stage": 0
        }), 400
    
    # Validate file type and save
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run unified two-stage prediction pipeline
            result = predict_leaf_disease(filepath)
            
            # Return result with appropriate HTTP status
            status_code = 200 if result.get("success", False) else 400
            return jsonify(result), status_code

        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Server error during prediction: {str(e)}",
                "stage": 0
            }), 500
    
    return jsonify({
        "success": False,
        "error": "Invalid file type. Please upload PNG, JPG, or JPEG.",
        "stage": 0
    }), 400


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors with custom template."""
    return render_template('404.html'), 404


if __name__ == "__main__":
    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT)
