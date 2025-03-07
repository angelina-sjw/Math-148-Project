import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from tqdm import tqdm

import torchvision.transforms as transforms


def resize_images(input_dir, output_dir, target_size=(224, 224)):
    """
    Resizes all images in the input directory and saves them in the output directory.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to save the resized images.
        target_size (tuple): Desired output size for images (default is 224x224).
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        # Create corresponding output directory
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                filename_without_ext = os.path.splitext(file)[0]
                output_path = os.path.join(output_subdir, filename_without_ext + ".jpg")
                
                try:
                    # Open and resize the image
                    with Image.open(input_path) as img:
                        img_resized = img.resize(target_size, Image.LANCZOS)
                        img_resized.save(output_path)
                        print(f"Processed: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")


def keep_existing_photos(df, dir):
    """
    Filters out missing images from the dataset.

    Args:
        df (DataFrame): DataFrame containing image metadata.
        dir (str): Directory where images are stored.

    Returns:
        DataFrame: A cleaned version of df containing only images that exist in the directory.
    """
    missing_images = []
    valid_indices = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        photo_id = row["photo_id"]
        image_path = f"{dir}/{photo_id}.jpg"

        if os.path.exists(image_path):
            valid_indices.append(idx)
        else:
            missing_images.append(
                {"index": idx, "photo_id": photo_id, "path": image_path}
            )

    clean_df = df.loc[valid_indices].reset_index(drop=True)

    return clean_df


def visualize_images(df, image_dir, label, label_value, num_samples=5):
    """
    Displays a sample of images from the dataset based on a specified label.

    Args:
        df (DataFrame): DataFrame containing image metadata.
        image_dir (str): Directory where images are stored.
        label (str): Column name for labels in df.
        label_value (any): Specific label value to filter images.
        num_samples (int): Number of images to display.
    """
    df = df[df[f'{label}'] == label_value]

    if df.empty:
        print(f"No images found for label: {label}")
        return

    sampled_images = df.sample(min(num_samples, len(df)))

    fig, axes = plt.subplots(1, len(sampled_images), figsize=(len(sampled_images) * 2, 2))

    if len(sampled_images) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, sampled_images.iterrows()):
        img_path = os.path.join(image_dir, f"{row['photo_id']}.jpg")

        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(row[label], fontsize=10, fontweight='bold')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def downsample_group(group, target_size):
    """
    Performs stratified downsampling of a group based on label distribution.

    Args:
        group (DataFrame): A group of data with label distributions.
        target_size (int): Desired number of samples after downsampling.

    Returns:
        DataFrame: A downsampled version of the group while maintaining label distribution.
    """
    label_counts = group['label'].value_counts(normalize=True)  
    sample_counts = (label_counts * target_size).astype(int)

    sampled_group = pd.concat([
        group[group['label'] == label].sample(n=min(count, len(group[group['label'] == label])), random_state=42)
        for label, count in sample_counts.items()
    ])

    return sampled_group


def preprocess_image(image_path):
    """
    Preprocesses an image for model input by resizing, normalizing, and converting to a tensor.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: (Original image, preprocessed image tensor)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    return image, input_tensor


def show_images(samples, title):
    """
    Displays a set of images with their confidence scores.

    Args:
        samples (list): A list of tuples (confidence, image tensor).
        title (str): Title of the visualization.
    """
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 5))
    for ax, (conf, img) in zip(axes, samples):
        img = img.cpu().numpy().transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_title(f"Conf: {conf:.4f}")
        ax.axis("off")
    plt.suptitle(title)
    plt.show()


def display_pseudolabels_based_on_confidence(df, min_conf, max_conf, image_dir, num_samples=5):
    filtered_df = df[(df["food101_confidences"] >= min_conf) & (df["food101_confidences"] <= max_conf)]
    sampled_df = filtered_df.sample(n=min(num_samples, len(filtered_df)))
    fig, axes = plt.subplots(1, len(sampled_df), figsize=(len(sampled_df) * 2, 2))

    if len(sampled_df) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, sampled_df.iterrows()):
        img_path = os.path.join(image_dir, f"{row['photo_id']}.jpg")

        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"{row['food101_predictions_decoded']} {row['food101_confidences']:.2f}", fontsize=10, fontweight='bold')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def unnormalize_image(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    unnorm_tensor = tensor.clone().detach().cpu()
    for c in range(3):
        unnorm_tensor[c] = unnorm_tensor[c] * std[c] + mean[c]
    unnorm_image = unnorm_tensor.permute(1, 2, 0).numpy()
    unnorm_image = np.clip(unnorm_image, 0, 1) 
    unnorm_image = (unnorm_image * 255).astype(np.uint8)
    return unnorm_image