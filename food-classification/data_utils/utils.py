import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms


def resize_images(input_dir, output_dir, target_size=(224, 224)):
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
    # Get the distribution of labels within this group
    label_counts = group['label'].value_counts(normalize=True)  # Get percentage representation
    sample_counts = (label_counts * target_size).astype(int)  # Calculate number of samples per label

    # Perform stratified sampling based on calculated sample counts
    sampled_group = pd.concat([
        group[group['label'] == label].sample(n=min(count, len(group[group['label'] == label])), random_state=42)
        for label, count in sample_counts.items()
    ])

    return sampled_group

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    return image, input_tensor