"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Provides explainability for CNN-based disease predictions by visualizing 
which regions of the input image the model focuses on during classification.

Production-grade implementation with automatic layer detection, robust error handling,
and configurable visualization options.

References:
    - Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via 
      Gradient-based Localization" (2017)
"""

import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from typing import Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import warnings


@dataclass
class GradCAMConfig:
    """Configuration for Grad-CAM generation."""
    
    output_dir: str = "static/gradcam"
    overlay_alpha: float = 0.4  # Heatmap contribution (40%)
    colormap: int = cv2.COLORMAP_JET
    generate_only_if_uncertain: bool = False
    uncertainty_threshold: float = 0.85  # Generate Grad-CAM if confidence < 0.85
    save_separate_heatmap: bool = False  # Save standalone heatmap in addition to overlay
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.overlay_alpha <= 1:
            raise ValueError(f"overlay_alpha must be between 0 and 1, got {self.overlay_alpha}")
        if not 0 <= self.uncertainty_threshold <= 1:
            raise ValueError(f"uncertainty_threshold must be between 0 and 1, got {self.uncertainty_threshold}")


def auto_detect_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Automatically detect the last convolutional layer in the model.
    
    Searches the model layers in reverse order and returns the name of the first
    Conv2D layer encountered. This is typically the optimal layer for Grad-CAM visualization.
    Handles nested models like transfer learning architectures.
    
    Args:
        model: TensorFlow/Keras model instance
        
    Returns:
        str: Name of the last convolutional layer
        
    Raises:
        ValueError: If no Conv2D layer is found in the model
    """
    # First, try to find EfficientNet/nested base model
    for layer in model.layers:
        layer_name = layer.name.lower()
        # Check if this is a transfer learning base model
        if any(name in layer_name for name in ['efficientnet', 'resnet', 'vgg', 'inception', 'mobilenet', 'xception']):
            # Use the output layer of the base model
            if hasattr(layer, 'output'):
                return layer.name
    
    # If no base model found, search for Conv2D layers in reverse
    for layer in reversed(model.layers):
        # Check for Conv2D layers
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        
        # Handle nested models (e.g., functional API, Sequential)
        if hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    # Return the parent layer name for nested models
                    return layer.name
    
    raise ValueError(
        "No Conv2D layer found in model. Grad-CAM requires a convolutional architecture. "
        f"Model layers: {[l.name for l in model.layers]}. "
        "Please specify the last_conv_layer_name manually."
    )


def load_and_preprocess_image(
    image_path: str,
    model: tf.keras.Model,
    preprocess_function: Optional[Callable] = None
) -> Tuple[np.ndarray, Image.Image]:
    """
    Load and preprocess image for Grad-CAM generation.
    Matches the exact preprocessing used during training (including center crop).
    
    Args:
        image_path: Path to the input image
        model: Model instance (used to extract input shape)
        preprocess_function: Custom preprocessing function (e.g., EfficientNet preprocessing).
                           If None, uses standard normalization to [0, 1].
        
    Returns:
        Tuple containing:
            - Preprocessed image array ready for model (shape: (1, H, W, 3))
            - Original PIL image for overlay visualization
            
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded or processed
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Load original image in RGB mode
        original_image = Image.open(image_path).convert("RGB")
        
        # Extract model input shape dynamically
        input_shape = model.input_shape
        if isinstance(input_shape, list):  # Handle multi-input models
            input_shape = input_shape[0]
        
        # Get target size (height, width) from model input shape
        # Shape format: (batch, height, width, channels)
        target_height = input_shape[1] if input_shape[1] is not None else 224
        target_width = input_shape[2] if input_shape[2] is not None else 224
        target_size = (target_width, target_height)
        
        # Center-crop to square (matches training preprocessing)
        width, height = original_image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        cropped_image = original_image.crop((left, top, right, bottom))
        
        # Resize to model input size
        resized_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array (0-255 range)
        img_array = np.array(resized_image, dtype=np.float32)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply preprocessing function if provided
        # For EfficientNet, this scales from [0,255] to approximately [-1,1]
        if preprocess_function is not None:
            img_array = preprocess_function(img_array)
        else:
            # Default: normalize to [0, 1]
            img_array = img_array / 255.0
        
        # CRITICAL FIX: EfficientNet preprocess_input may not work in some TF versions
        # If the array still has values > 1, manually apply EfficientNet preprocessing
        # EfficientNet uses mode='torch': scale to [0,1] then normalize with ImageNet stats
        if img_array.max() > 1.0:
            # Scale to [0, 1]
            img_array = img_array / 255.0
            # Apply ImageNet normalization (mode='torch')
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std
        
        return img_array, original_image
        
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {str(e)}")


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: str,
    pred_index: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Compute Grad-CAM heatmap for a given image and model.
    
    Uses gradient-weighted class activation mapping to highlight regions of the image
    that contribute most to the model's prediction.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array (shape: (1, H, W, 3))
        last_conv_layer_name: Name of the last convolutional layer to use for Grad-CAM
        pred_index: Target class index. If None, uses the predicted class (argmax).
        
    Returns:
        Tuple containing:
            - Heatmap array (shape: (H, W)) with values in [0, 1]
            - Predicted class index
            
    Raises:
        ValueError: If the specified layer is not found or is invalid
    """
    # Build gradient model
    try:
        # Get the last convolutional layer and the model output
        last_conv_layer = model.get_layer(last_conv_layer_name)
        
        # For models with nested Functional layers (like EfficientNet inside Sequential),
        # we need to create the grad_model carefully
        # The issue is that when last_conv_layer is from a nested Functional model,
        # using model.inputs may not work correctly
        
        # Get the model's input (handle both single and multiple inputs)
        model_inputs = model.inputs if isinstance(model.inputs, list) else [model.inputs]
        
        # Create a model that maps input image to:
        # 1. The activations of the last conv layer
        # 2. The model's prediction output
        grad_model = tf.keras.models.Model(
            inputs=model_inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        
    except Exception as e:
        raise ValueError(
            f"Error building gradient model with layer '{last_conv_layer_name}': {str(e)}"
        )
    
    # Compute gradients
    # Create an intermediate model only for the conv layer outputs
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        # Use model.inputs[0] if available, otherwise use the input directly
        intermediate_model = tf.keras.Sequential()
        # This is problematic with Functional models, so we'll do it without sub-models
    except Exception as e:
        pass
    
    with tf.GradientTape() as tape:
        # Convert to tensor if needed (GradientTape needs tf.Tensor, not ndarray)
        # Ensure it's float32, not float64
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        tape.watch(img_tensor)
        
        # Get predictions from the full model
        model_predictions = model(img_tensor, training=False)
        if isinstance(model_predictions, list):
            model_predictions = model_predictions[0]
        
        # Get the score for the predicted class
        if pred_index is None:
            pred_index = tf.argmax(model_predictions[0]).numpy()
        
        class_channel = model_predictions[:, pred_index]
    
    # Compute gradients of the predicted class score with respect to the input
    input_grads = tape.gradient(class_channel, img_tensor)
    
    # Now we need to compute gradients with respect to the conv layer
    # Create a new gradient tape for conv layer activation
    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        # Get conv layer activations using a partial model
        # Use the layer directly from the original model
        last_conv_layer = model.get_layer(last_conv_layer_name)
        
        # Build a model that extracts conv layer output
        # For nested models, we need to traverse carefully
        # Create model from input to conv layer output
        input_layer = model.input
        conv_output = last_conv_layer.output
        
        # Create submodel using functional API
        try:
            # This might fail with nested Functional models
            partial_model = tf.keras.Model(inputs=input_layer, outputs=conv_output)
            conv_outputs = partial_model(img_tensor, training=False)
        except:
            # Fallback: manually get layer outputs
            # Get all intermediate activations
            def get_conv_output(inp):
                # Use model.predict on intermediate layer
                intermediate_layer_model = tf.keras.Model(inputs=model.inputs, 
                                                         outputs=last_conv_layer.output)
                return intermediate_layer_model.predict(inp, verbose=0)
            
            conv_outputs = tf.constant(get_conv_output(img_tensor.numpy()))
        
        # Handle case where outputs are lists (common in Functional API)
        if isinstance(conv_outputs, list):
            conv_outputs = conv_outputs[0]
        
        # Get the score for the predicted class
        class_channel = model_predictions[:, pred_index]
    
    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)
    
    
    # Global average pooling of gradients across spatial dimensions (H, W)
    # This gives us the import ance weight for each feature map channel
    # Shape: (batch, channels) after reduction over (height, width)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Get the activation map values
    conv_outputs = conv_outputs[0]  # Remove batch dimension
    
    # Multiply each channel in the feature map by how important it is
    # with regard to the predicted class, then sum all channels
    # This gives us the heatmap
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)  # Remove single-dimensional entries
    
    # Apply ReLU to remove negative values (we only want positive contributions)
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalize heatmap to [0, 1] range
    heatmap_max = tf.reduce_max(heatmap)
    if heatmap_max == 0:
        warnings.warn("Heatmap contains all zeros. The model may not be using this layer for prediction.")
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = heatmap / heatmap_max
    
    return heatmap.numpy(), int(pred_index)


def create_heatmap_overlay(
    heatmap: np.ndarray,
    original_image: Image.Image,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Create an overlay of the Grad-CAM heatmap on the original image.
    
    Args:
        heatmap: Normalized heatmap array (values in [0, 1])
        original_image: Original PIL image
        alpha: Heatmap contribution to overlay (0.4 = 40% heatmap, 60% image)
        colormap: OpenCV colormap to apply to heatmap
        
    Returns:
        Overlaid image as numpy array (RGB, uint8)
    """
    # Resize heatmap to match original image size
    img_width, img_height = original_image.size
    heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
    
    # Convert heatmap to uint8 [0, 255] range
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap (converts to BGR)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert heatmap from BGR to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Convert original image to numpy array
    original_array = np.array(original_image, dtype=np.uint8)
    
    # Blend heatmap with original image
    # overlay = alpha * heatmap + (1 - alpha) * original
    overlay = cv2.addWeighted(
        heatmap_colored, alpha,
        original_array, 1 - alpha,
        0
    )
    
    return overlay


def save_gradcam_image(
    overlay: np.ndarray,
    output_dir: str,
    prefix: str = "gradcam"
) -> str:
    """
    Save Grad-CAM overlay image to disk.
    
    Args:
        overlay: Overlay image as numpy array (RGB)
        output_dir: Directory to save the image
        prefix: Filename prefix
        
    Returns:
        Relative path to the saved image
        
    Raises:
        IOError: If image cannot be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    try:
        # Convert to PIL Image and save
        overlay_image = Image.fromarray(overlay, mode='RGB')
        overlay_image.save(filepath, quality=95, optimize=True)
        
        # Return relative path (for web serving)
        relative_path = filepath.replace("\\", "/")
        return relative_path
        
    except Exception as e:
        raise IOError(f"Failed to save Grad-CAM image to {filepath}: {str(e)}")


def generate_gradcam(
    model: tf.keras.Model,
    image_path: str,
    preprocess_function: Optional[Callable] = None,
    last_conv_layer_name: Optional[str] = None,
    config: Optional[GradCAMConfig] = None
) -> Dict[str, Any]:
    """
    Generate Grad-CAM visualization for a given image and model prediction.
    
    This is the main entry point for Grad-CAM generation. It orchestrates all steps:
    1. Load and preprocess image
    2. Auto-detect last conv layer (if not specified)
    3. Compute Grad-CAM heatmap
    4. Create overlay visualization
    5. Save result to disk
    
    Args:
        model: Trained Keras model (must be CNN-based)
        image_path: Path to input image
        preprocess_function: Preprocessing function used during training.
                           Should match the exact preprocessing used when training the model.
                           For EfficientNet: tf.keras.applications.efficientnet.preprocess_input
        last_conv_layer_name: Name of the last convolutional layer. If None, auto-detects.
        config: GradCAMConfig instance for customization. Uses defaults if None.
        
    Returns:
        Dictionary containing:
            - success (bool): Whether Grad-CAM generation succeeded
            - heatmap_path (str): Path to saved overlay image
            - prediction_index (int): Predicted class index
            - confidence (float): Model confidence for predicted class
            - error (str): Error message if success=False
            
    Raises:
        ValueError: If model is invalid or image cannot be processed
        FileNotFoundError: If image file doesn't exist
    """
    # Use default config if none provided
    if config is None:
        config = GradCAMConfig()
    
    try:
        # STEP 1: Load and preprocess image
        img_array, original_image = load_and_preprocess_image(
            image_path, model, preprocess_function
        )
        
        # STEP 2: Get prediction and confidence
        predictions = model.predict(img_array, verbose=0)
        pred_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][pred_index])
        
        # STEP 3: Check if we should generate Grad-CAM based on uncertainty
        if config.generate_only_if_uncertain and confidence >= config.uncertainty_threshold:
            return {
                "success": False,
                "heatmap_path": None,
                "prediction_index": pred_index,
                "confidence": confidence,
                "error": f"Skipped: Confidence {confidence:.2%} >= threshold {config.uncertainty_threshold:.2%}",
                "skipped": True
            }
        
        # STEP 4: Auto-detect last conv layer if not provided
        if last_conv_layer_name is None:
            last_conv_layer_name = auto_detect_last_conv_layer(model)
        
        # STEP 5: Compute Grad-CAM heatmap
        heatmap, pred_index = compute_gradcam_heatmap(
            model, img_array, last_conv_layer_name, pred_index
        )
        
        # STEP 6: Create overlay visualization
        overlay = create_heatmap_overlay(
            heatmap, original_image, 
            alpha=config.overlay_alpha,
            colormap=config.colormap
        )
        
        # STEP 7: Save overlay image
        heatmap_path = save_gradcam_image(
            overlay, config.output_dir, prefix="gradcam"
        )
        
        # STEP 8: Optionally save standalone heatmap
        heatmap_only_path = None
        if config.save_separate_heatmap:
            # Resize and colorize heatmap
            img_width, img_height = original_image.size
            heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, config.colormap)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            heatmap_only_path = save_gradcam_image(
                heatmap_colored, config.output_dir, prefix="gradcam_heatmap_only"
            )
        
        # Return success result
        return {
            "success": True,
            "heatmap_path": heatmap_path,
            "heatmap_only_path": heatmap_only_path,
            "prediction_index": pred_index,
            "confidence": confidence,
            "last_conv_layer_used": last_conv_layer_name,
            "error": None
        }
        
    except FileNotFoundError as e:
        return {
            "success": False,
            "heatmap_path": None,
            "prediction_index": None,
            "confidence": None,
            "error": f"Image file error: {str(e)}"
        }
    
    except ValueError as e:
        return {
            "success": False,
            "heatmap_path": None,
            "prediction_index": None,
            "confidence": None,
            "error": f"Model or processing error: {str(e)}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "heatmap_path": None,
            "prediction_index": None,
            "confidence": None,
            "error": f"Unexpected error during Grad-CAM generation: {str(e)}"
        }


# Convenience function for quick testing
def test_gradcam(model_path: str, image_path: str, last_conv_layer_name: Optional[str] = None):
    """
    Quick test function for Grad-CAM generation.
    
    Args:
        model_path: Path to saved Keras model
        image_path: Path to test image
        last_conv_layer_name: Optional layer name
        
    Returns:
        Grad-CAM generation result dictionary
    """
    model = tf.keras.models.load_model(model_path)
    
    # Use EfficientNet preprocessing by default
    preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
    
    result = generate_gradcam(
        model=model,
        image_path=image_path,
        preprocess_function=preprocess_fn,
        last_conv_layer_name=last_conv_layer_name
    )
    
    if result["success"]:
        print(f"✓ Grad-CAM generated successfully")
        print(f"  Saved to: {result['heatmap_path']}")
        print(f"  Predicted class: {result['prediction_index']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Layer used: {result['last_conv_layer_used']}")
    else:
        print(f"✗ Grad-CAM generation failed")
        print(f"  Error: {result['error']}")
    
    return result


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) >= 3:
        model_path = sys.argv[1]
        image_path = sys.argv[2]
        layer_name = sys.argv[3] if len(sys.argv) > 3 else None
        
        test_gradcam(model_path, image_path, layer_name)
    else:
        print("Usage: python gradcam.py <model_path> <image_path> [last_conv_layer_name]")
        print("\nExample:")
        print("  python gradcam.py models/leaf_model2.keras test_image.jpg")
