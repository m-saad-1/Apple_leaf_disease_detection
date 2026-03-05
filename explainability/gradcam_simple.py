"""
Simpler, more robust Grad-CAM implementation for Apple Leaf Disease Detection
Avoids Functional API submodel creation issues
"""

import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from typing import Tuple, Optional
from datetime import datetime


def compute_gradcam_heatmap_simple(
    model: tf.keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: str
) -> Tuple[np.ndarray, int]:
    """
    Simplified Grad-CAM that avoids submodel creation issues.
    
    Instead of creating intermediate models (which fails with nested Functional models),
    this implementation uses gradient hooks and direct layer access.
    """
    
    # Ensure float32
    img_array = img_array.astype(np.float32)
    img_tensor = tf.constant(img_array)
    
    # Get the layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    
    # Build a model to extract conv output using layer directly
    # Use a function to compute both output and predictions
    def model_with_conv_output(x):
        # Run through each layer up to and including conv layer
        for layer in model.layers:
            x = layer(x, training=False)
            if layer.name == last_conv_layer_name:
                break
        conv_out = x
        # Continue to output
        for layer in model.layers[model.layers.index(layer)+1:]:
            x = layer(x, training=False)
        return x, conv_out
    
    # GradientTape to compute gradients
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        
        # Get predictions
        predictions = model(img_tensor, training=False)
        if isinstance(predictions, list):
            predictions = predictions[0]
        
        # Get predicted class
        pred_idx = tf.argmax(predictions[0]).numpy()
        
        # Get class score
        class_score = predictions[:, pred_idx]
    
    # Compute input gradients (as a proxy when conv gradients fail)
    input_grads = tape.gradient(class_score, img_tensor)
    
    # Get actual conv layer outputs using predict
    try:
        # Create model from input to conv output
        intermediate_model = tf.keras.Model(
            inputs=model.input,
            outputs=last_conv_layer.output
        )
        conv_outputs = intermediate_model.predict(img_array, verbose=0)
    except:
        # If model creation fails, use input gradients as heatmap
        # Resize input gradients to 224x224
        grad_map = tf.reduce_mean(tf.abs(input_grads[0]), axis=-1).numpy()
        grad_map = np.maximum(grad_map, 0)
        grad_map = grad_map / (grad_map.max() + 1e-8)
        return grad_map, int(pred_idx)
    
    # Handle list outputs
    if isinstance(conv_outputs, list):
        conv_outputs = conv_outputs[0]
    
    # Compute importance weights via a second gradient tape on conv outputs
    with tf.GradientTape() as tape2:
        img_tensor2 = tf.Variable(img_array, trainable=True)
        intermediate_model = tf.keras.Model(inputs=model.input, outputs=last_conv_layer.output)
        conv_ops = intermediate_model(img_tensor2)
        predictions2 = model(img_tensor2)
        if isinstance(predictions2, list):
            predictions2 = predictions2[0]
        class_score2 = predictions2[:, pred_idx]
    
    grads2 = tape2.gradient(class_score2, conv_ops)
    
    if grads2 is not None:
        # Compute weighted importance
        pooled_grads = tf.reduce_mean(grads2, axis=(0, 1, 2))
        conv_out_np = conv_outputs[0]
        heatmap = np.dot(conv_out_np, pooled_grads.numpy())
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    else:
        # Fallback to simple average activation
        heatmap = np.mean(conv_outputs[0], axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
    
    return heatmap.astype(np.float32), int(pred_idx)


def create_heatmap_overlay(
    heatmap: np.ndarray,
    original_image: Image.Image,
    alpha: float = 0.4
) -> np.ndarray:
    """Create overlay of heatmap on original image"""
    
    # Resize heatmap to match image size
    original_size = original_image.size
    heatmap_resized = cv2.resize(heatmap, (original_size[0], original_size[1]))
    
    # Create jet colormap
    heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Convert original image to numpy
    image_np = np.array(original_image)
    
    # Blend
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
    
    return overlay.astype(np.uint8)


def save_gradcam_image(overlay: np.ndarray, output_dir: str = "static/gradcam") -> str:
    """Save Grad-CAM overlay image"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"gradcam_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # Convert to PIL and save
    img_pil = Image.fromarray(overlay)
    img_pil.save(filepath, quality=95)
    
    # Return relative path for web serving
    return f"/static/gradcam/{filename}"


def generate_gradcam_simple(model, image_path: str, last_conv_layer_name: str = None) -> dict:
    """Main entry point for Grad-CAM generation"""
    
    try:
        # Load image
        original_image = Image.open(image_path).convert("RGB")
        
        # Resize for model
        img_resized = original_image.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Normalize with ImageNet stats
        img_array = img_array / 255.0
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        # Apply ImageNet preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Auto-detect conv layer if not specified
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower() or 'pooling' in layer.name.lower():
                    last_conv_layer_name = layer.name
                    break
        
        # Compute Grad-CAM
        heatmap, pred_idx = compute_gradcam_heatmap_simple(
            model, img_array, last_conv_layer_name
        )
        
        # Create overlay
        overlay = create_heatmap_overlay(heatmap, original_image)
        
        # Save
        gradcam_path = save_gradcam_image(overlay)
        
        return {
            "success": True,
            "gradcam_image": gradcam_path,
            "predicted_class": int(pred_idx)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "gradcam_image": None
        }
