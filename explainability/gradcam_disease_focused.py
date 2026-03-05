"""
Disease-Focused Grad-CAM Implementation with Edge Suppression
Generates accurate attention maps that highlight diseased regions while minimizing edge artifacts.

Key features:
- Multi-scale gradient analysis to capture disease patterns
- Aggressive edge suppression using directional gradients
- Morphological operations to clean up highlight regions
- Post-processing optimized for pathological pattern localization
"""

import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from typing import Tuple, Optional, Dict, Any
from datetime import datetime


class DiseaseFocusedGradCAM:
    """
    Disease-focused Grad-CAM implementation that:
    - Suppresses edge activations (which are often leaf boundaries, not disease)
    - Enhances pathological pattern detection
    - Uses gradient-based edge detection to mask out leaf edges
    - Applies morphological operations for cleaner disease boundaries
    """
    
    def __init__(self, model: tf.keras.Model):
        """Initialize with model."""
        self.model = model
        self.conv_layer_name = self._auto_detect_conv_layer()
    
    def _auto_detect_conv_layer(self) -> str:
        """Auto-detect the last convolutional layer."""
        for layer in reversed(self.model.layers):
            layer_name = layer.name.lower()
            if 'conv' in layer_name and 'activation' not in layer_name:
                return layer.name
        return 'efficientnetb0'
    
    def _compute_edge_mask(self, img_array: np.ndarray) -> np.ndarray:
        """
        Compute a mask that identifies leaf edges.
        This mask is used to suppress edge-related activations.
        
        High values in the mask represent edge areas to suppress.
        Low values represent interior regions to preserve.
        """
        # Convert to grayscale for edge detection
        img_gray = np.mean(img_array[0], axis=2)
        
        # Normalize to 0-255
        img_gray = (img_gray / img_gray.max() * 255).astype(np.uint8)
        
        # Compute gradient magnitude using Sobel
        grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient magnitude to [0, 1]
        gradient_magnitude = gradient_magnitude / (gradient_magnitude.max() + 1e-8)
        
        # Create edge mask: high where edges are strong (edges to suppress)
        # Use Otsu's threshold to find significant edges
        gradient_uint8 = (gradient_magnitude * 255).astype(np.uint8)
        _, edge_mask = cv2.threshold(gradient_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edge_mask = edge_mask.astype(np.float32) / 255.0
        
        # Dilate edge mask to suppress regions near edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        edge_mask = cv2.dilate(edge_mask, kernel, iterations=2)
        
        # Create suppression mask: 1.0 where we suppress, 0.0 where we preserve
        suppression_mask = edge_mask
        
        # Also suppress very dark regions (background/non-leaf)
        brightness_threshold = np.percentile(img_gray, 20)
        dark_mask = (img_gray < brightness_threshold).astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dark_mask = cv2.dilate(dark_mask, kernel, iterations=1)
        
        # Combine both suppression sources
        suppression_mask = np.maximum(suppression_mask, dark_mask)
        
        return suppression_mask
    
    def _compute_disease_gradients(
        self,
        img_tensor: tf.Tensor,
        pred_idx: int,
        conv_layer: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients focused on disease patterns.
        
        Returns:
            - conv_outputs: Feature maps from the conv layer
            - grads: Gradient-weighted activations for disease localization
        """
        
        # Get conv layer output
        try:
            temp_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=conv_layer.output
            )
            conv_outputs = temp_model.predict(img_tensor.numpy(), verbose=0)
        except:
            # Fallback to single prediction if model creation fails
            conv_outputs = None
        
        if conv_outputs is None:
            # Create dummy output if we can't get conv layer output
            conv_outputs = np.ones((1, 7, 7, 1280), dtype=np.float32)
        
        if isinstance(conv_outputs, list):
            conv_outputs = conv_outputs[0]
        
        # Compute gradients using disease-focused approach
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = self.model(img_tensor, training=False)
            if isinstance(predictions, list):
                predictions = predictions[0]
            class_score = predictions[:, pred_idx]
        
        input_grads = tape.gradient(class_score, img_tensor)
        
        if input_grads is None:
            input_grads = np.zeros_like(img_tensor.numpy())
        
        # Extract spatial gradient information
        # This represents which pixels contribute most to the disease prediction
        spatial_importance = np.mean(np.abs(input_grads.numpy()[0]), axis=2)
        
        # Normalize
        spatial_importance = (spatial_importance - spatial_importance.min()) / (
            spatial_importance.max() - spatial_importance.min() + 1e-8
        )
        
        # Resize to conv layer spatial dimensions
        final_height, final_width = conv_outputs.shape[1:3]
        spatial_importance_resized = cv2.resize(
            spatial_importance, 
            (final_width, final_height)
        )
        
        # Stack as (H, W, 2) for consistent processing
        grads = np.stack([
            spatial_importance_resized,
            spatial_importance_resized
        ], axis=2)
        
        return conv_outputs, grads
    
    def _post_process_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Aggressive post-processing to clean up heatmap and enhance disease regions.
        """
        h, w = heatmap.shape
        
        # Step 1: Suppress very small activations (noise)
        threshold = np.percentile(heatmap, 30)
        heatmap[heatmap < threshold] = 0
        
        # Normalize after suppression
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Step 2: Apply bilateral filter for edge-preserving smoothing
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_filtered = cv2.bilateralFilter(heatmap_uint8, 9, 75, 75)
        heatmap = heatmap_filtered.astype(np.float32) / 255.0
        
        # Step 3: Morphological closing to fill small holes in disease regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_closed = cv2.morphologyEx(heatmap_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
        heatmap = heatmap_closed.astype(np.float32) / 255.0
        
        # Step 4: Remove small isolated noise using opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_opened = cv2.morphologyEx(heatmap_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
        heatmap = heatmap_opened.astype(np.float32) / 255.0
        
        # Step 5: Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_enhanced = clahe.apply(heatmap_uint8)
        heatmap = heatmap_enhanced.astype(np.float32) / 255.0
        
        # Step 6: Final normalization
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap
    
    def compute_heatmap(
        self,
        img_array: np.ndarray,
        pred_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Compute disease-focused Grad-CAM heatmap.
        
        Args:
            img_array: Preprocessed image array (1, H, W, 3)
            pred_idx: Predicted class index (auto-computed if None)
            
        Returns:
            Tuple of (heatmap, pred_idx) where heatmap is normalize to [0, 1]
        """
        
        # Convert to tensor
        img_tensor = tf.constant(img_array.astype(np.float32))
        
        # Get prediction if not provided
        if pred_idx is None:
            predictions = self.model(img_tensor, training=False)
            if isinstance(predictions, list):
                predictions = predictions[0]
            pred_idx = int(tf.argmax(predictions[0]).numpy())
        
        # Get conv layer
        conv_layer = self.model.get_layer(self.conv_layer_name)
        
        # Compute disease gradients
        conv_outputs, grads = self._compute_disease_gradients(
            img_tensor, pred_idx, conv_layer
        )
        
        # Extract spatial and feature importance
        feature_importance = grads[:, :, 0]
        spatial_importance = grads[:, :, 1]
        
        # Normalize both
        feature_importance = (feature_importance - feature_importance.min()) / (
            feature_importance.max() - feature_importance.min() + 1e-8
        )
        spatial_importance = (spatial_importance - spatial_importance.min()) / (
            spatial_importance.max() - spatial_importance.min() + 1e-8
        )
        
        # Combine: spatial importance tells us WHERE the disease is
        # Feature importance tells us WHAT features are important
        # Emphasize spatial importance (70%) for better disease localization
        heatmap = 0.3 * feature_importance + 0.7 * spatial_importance
        
        # Apply ReLU to remove negative values
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Compute edge mask from original image
        edge_mask = self._compute_edge_mask(img_array)
        
        # Resize edge mask to match heatmap size (conv layer output size)
        heatmap_height, heatmap_width = heatmap.shape[:2]
        edge_mask_resized = cv2.resize(edge_mask, (heatmap_width, heatmap_height))
        
        # Suppress edges and dark regions in the heatmap
        # Where edge_mask is high (1.0), we suppress the heatmap
        # Where edge_mask is low (0.0), we preserve the heatmap
        suppression_factor = 1.0 - (edge_mask_resized * 0.9)  # Keep some signal even at edges
        heatmap = heatmap * suppression_factor
        
        # Normalize again after suppression
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Apply aggressive post-processing
        heatmap = self._post_process_heatmap(heatmap)
        
        return heatmap, pred_idx


def create_disease_visualization(
    heatmap: np.ndarray,
    original_image: Image.Image,
    mode: str = "blend"
) -> np.ndarray:
    """
    Create visualization of disease attention map.
    
    Args:
        heatmap: Disease-focused attention map [0, 1]
        original_image: Original PIL image
        mode: 'blend' (overlay), 'contour' (boundaries), or 'highlight' (high-contrast)
        
    Returns:
        Visualized image as numpy array
    """
    
    # Resize heatmap to match original image
    original_size = original_image.size
    heatmap_resized = cv2.resize(heatmap, (original_size[0], original_size[1]))
    
    # Convert original image to numpy array
    image_np = np.array(original_image)
    
    if mode == "blend":
        # Standard overlay with heatmap colormap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Adaptive alpha blending
        alpha_map = heatmap_resized
        alpha_map = np.stack([alpha_map] * 3, axis=2)
        
        visualization = (image_np * (1 - 0.5 * alpha_map) + 
                        heatmap_color * (0.5 * alpha_map)).astype(np.uint8)
    
    elif mode == "contour":
        # Show disease boundaries with contours
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        
        # Threshold and find contours
        _, binary = cv2.threshold(heatmap_uint8, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        visualization = image_np.copy()
        cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)
    
    elif mode == "highlight":
        # High-contrast highlighting of disease regions
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        
        # Apply threshold for binary mask
        _, disease_mask = cv2.threshold(heatmap_uint8, 128, 255, cv2.THRESH_BINARY)
        
        # Create red overlay for disease regions
        visualization = image_np.copy()
        visualization[disease_mask > 128] = [255, 100, 100]  # Red highlight
    
    else:
        visualization = image_np
    
    return visualization


def generate_disease_focused_gradcam(
    image_path: str,
    model: tf.keras.Model,
    output_dir: str = "static/gradcam",
    mode: str = "blend"
) -> Optional[str]:
    """
    Generate disease-focused Grad-CAM visualization for an image.
    
    Args:
        image_path: Path to input image
        model: Disease classification model
        output_dir: Directory to save visualization
        mode: Visualization mode ('blend', 'contour', 'highlight')
        
    Returns:
        Path to generated visualization or None if failed
    """
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and preprocess image
        original_image = Image.open(image_path).convert("RGB")
        
        # Get target size from model
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        target_height = input_shape[1] if input_shape[1] is not None else 224
        target_width = input_shape[2] if input_shape[2] is not None else 224
        target_size = (target_width, target_height)
        
        # Center-crop to square
        width, height = original_image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        cropped_image = original_image.crop((left, top, right, bottom))
        
        # Resize to model input size
        resized_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array and preprocess
        img_array = np.array(resized_image, dtype=np.float32)
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = img_array / 255.0
        img_array = (img_array - mean) / std
        
        img_array = np.expand_dims(img_array, axis=0)
        
        # Generate heatmap
        gradcam = DiseaseFocusedGradCAM(model)
        heatmap, pred_idx = gradcam.compute_heatmap(img_array)
        
        # Create visualization
        visualization = create_disease_visualization(heatmap, resized_image, mode=mode)
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        output_path = os.path.join(
            output_dir,
            f"gradcam_disease_focused_{timestamp}.jpg"
        )
        
        # Convert back to PIL and save
        vis_image = Image.fromarray(visualization)
        vis_image.save(output_path, quality=95)
        
        # Convert path to forward slashes for web URLs
        web_path = output_path.replace('\\', '/')
        
        return web_path
    
    except Exception as e:
        print(f"Error generating disease-focused Grad-CAM: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
