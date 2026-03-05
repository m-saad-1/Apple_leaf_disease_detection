"""
Explainability Module for Apple Leaf Disease Detection
Provides interpretability tools for understanding model predictions.

Includes disease-focused Grad-CAM implementation:
- generate_disease_focused_gradcam: disease-localized visualization with edge suppression
- DiseaseFocusedGradCAM: Main class for heatmap generation
"""

from .gradcam_disease_focused import (
    generate_disease_focused_gradcam,
    DiseaseFocusedGradCAM,
    create_disease_visualization
)

__all__ = [
    'generate_disease_focused_gradcam',
    'DiseaseFocusedGradCAM',
    'create_disease_visualization'
]
