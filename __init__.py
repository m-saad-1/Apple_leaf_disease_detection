"""
Apple Leaf Disease Detection System
A production-ready two-stage deep learning classification system for detecting apple leaf diseases.

Version: 2.0
Status: Production Ready

Two-Stage Pipeline:
- Stage 1: Detect if image is an apple leaf and classify health status
- Stage 2: Identify specific disease type for diseased leaves

Usage:
    from predict import predict_leaf_disease
    result = predict_leaf_disease("path/to/image.jpg")
"""

__version__ = "2.0.0"
__author__ = "Apple Leaf Detection Team"
__description__ = "Two-Stage Apple Leaf Disease Classification System"

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
