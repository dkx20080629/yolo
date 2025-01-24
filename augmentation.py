import numpy as np
import torch
from torchvision import transforms
import os
from PIL import Image
import random

# Define augmentation transforms
transform_list = [
    transforms.RandomHorizontalFlip(p=1.0),          # Horizontal flip
    transforms.RandomVerticalFlip(p=1.0),            # Vertical flip
    transforms.RandomRotation(degrees=45),           # Random rotation (±45 degrees)
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # Brightness, contrast, saturation
    transforms.RandomResizedCrop(size=(180, 180), scale=(0.8, 1.0)),  # Random crop and resize
]

# Combine augmentations into a sequential transform
augmentations = transforms.Compose(transform_list)

# Directory to save augmented images and labels
output_dir_images = "augmented_images"
output_dir_labels = "augmented_labels"
os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_labels, exist_ok=True)

# Function to add Gaussian noise to an image
def add_gaussian_noise(image, mean=0, std=0.1):
    np_image = np.array(image) / 255.0  # Normalize to [0, 1]
    noise = np.random.normal(mean, std, np_image.shape)  # Add noise
    noisy_image = np.clip(np_image + noise, 0, 1)  # Ensure values are within [0, 1]
    noisy_image = (noisy_image * 255).astype(np.uint8)  # Rescale to [0, 255]
    return Image.fromarray(noisy_image)

# Function to add salt-and-pepper noise to an image
def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    np_image = np.array(image)  # Convert image to numpy array
    total_pixels = np_image.size
    num_salt = int(salt_prob * total_pixels)
    num_pepper = int(pepper_prob * total_pixels)
    
    # Add salt noise
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in np_image.shape]
    np_image[salt_coords[0], salt_coords[1], :] = 255  # Salt = white
    
    # Add pepper noise
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in np_image.shape]
    np_image[pepper_coords[0], pepper_coords[1], :] = 0  # Pepper = black
    
    return Image.fromarray(np_image)

# Function to read YOLO-style annotations and adjust based on augmentation
def read_and_augment_label(label_path, transform, image_size):
    # Read the label (YOLO format: class_id x_center y_center width height)
    with open(label_path, 'r') as file:
        lines = file.readlines()

    augmented_labels = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
        
        # Apply transformations to bounding boxes
        if isinstance(transform, transforms.RandomHorizontalFlip):
            x_center = 1 - x_center  # Flip x_center horizontally
        elif isinstance(transform, transforms.RandomVerticalFlip):
            y_center = 1 - y_center  # Flip y_center vertically
        elif isinstance(transform, transforms.RandomRotation):
            # For simplicity, this example doesn't handle rotation, but you can adjust bounding boxes here
            pass
        elif isinstance(transform, transforms.RandomResizedCrop):
            # Handle crop/resize transformation, we assume it doesn't affect the bounding box much for this example
            pass
        
        augmented_labels.append([class_id, x_center, y_center, width, height])

    return augmented_labels

# Function to apply augmentations to both image and label and save
def save_augmented_images_and_labels(image_path, label_path, num_augmentations=10, noise_type='gaussian'):
    # Load the input image
    original_image = Image.open(image_path).convert("RGB")
    
    # Read and augment the label
    image_size = original_image.size  # (width, height)
    augmented_labels = read_and_augment_label(label_path, augmentations, image_size)

    # Apply augmentations and save
    for i in range(num_augmentations):
        augmented_image = augmentations(original_image)

        # Add noise to the augmented image
        if noise_type == 'gaussian':
            noisy_image = add_gaussian_noise(augmented_image)
        elif noise_type == 'salt_and_pepper':
            noisy_image = add_salt_and_pepper_noise(augmented_image)
        else:
            noisy_image = augmented_image  # No noise if no type is specified
        
        # Save augmented image
        save_path_image = os.path.join(output_dir_images, f"augmented_image_{i + 1}.jpg")
        save_path_label = os.path.join(output_dir_labels, f"augmented_label_{i + 1}.txt")
        
        noisy_image.save(save_path_image)
        
        # Save the augmented labels (in YOLO format)
        with open(save_path_label, 'w') as label_file:
            for label in augmented_labels:
                label_file.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")
        
        print(f"Saved: {save_path_image} and {save_path_label}")

# Example usage: Replace with your image and label paths
input_image = r"D:\Desktop\yolo\high_conf_images\images\class_1\20250124_132656897840-0.83.jpg"  # Provide the path to your input image
input_label = r"D:\Desktop\yolo\high_conf_images\labels\class_1\20250124_132656897840-0.83.txt"  # Provide the path to your label file

# Choose noise type: 'gaussian' or 'salt_and_pepper'
save_augmented_images_and_labels(input_image, input_label, num_augmentations=15, noise_type='gaussian')
