import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def create_augmentation_grid(image_path, output_dir):
    """
    Creates a 2x3 grid demonstrating various data augmentations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        # Create a mock color block if no image is provided
        img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    else:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))

    # Define augmentations
    aug_images = []
    titles = []

    # 1. Original
    aug_images.append(img)
    titles.append("Original")

    # 2. Rotation
    aug_images.append(img.rotate(45))
    titles.append("Rotation (45°)")

    # 3. Horizontal Flip
    aug_images.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    titles.append("Horizontal Flip")

    # 4. Brightness Jitter
    enhancer = ImageEnhance.Brightness(img)
    aug_images.append(enhancer.enhance(1.5)) # 1.5x brightness
    titles.append("Brightness Jitter")

    # 5. Gaussian Noise
    img_array = np.array(img)
    noise = np.random.normal(0, 25, img_array.shape) # mean=0, std=25
    noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    aug_images.append(Image.fromarray(noisy_img_array))
    titles.append("Gaussian Noise")

    # 6. Gaussian Blur
    aug_images.append(img.filter(ImageFilter.GaussianBlur(radius=2)))
    titles.append("Gaussian Blur")

    # Plot 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (ax, aug_img, title) in enumerate(zip(axes, aug_images, titles)):
        ax.imshow(aug_img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'augmentation_grid.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Saved augmentation grid to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmentation grid figure")
    parser.add_argument('--image', type=str, default='../data/sample_leaf.jpg', help="Path to sample image")
    parser.add_argument('--output', type=str, default='../outputs', help="Output directory")
    
    args = parser.parse_args()
    create_augmentation_grid(args.image, args.output)
