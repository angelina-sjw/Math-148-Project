import torch
from torch.utils.data import Dataset, Subset

import torchvision.transforms as transforms
from torchvision.transforms import RandAugment

from PIL import Image
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


class PhotoLabelDataset(Dataset):
    """Dataset for loading images and their corresponding labels from a dataframe."""
    def __init__(self, df, dir, label, transform=None):
        self.df = df
        self.dir = dir
        self.label = label
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Retrieves an image and its label based on the index."""
        if isinstance(idx, slice):
            return PhotoLabelDataset(self.df[idx], self.dir, self.label, transform=self.transform)

        row = self.df.iloc[idx]

        photo_id = row["photo_id"]

        try:
            img = self._load_image(photo_id)
        except Exception as e:
            print(f"Warning: Could not load {photo_id} ({e})")

        label = torch.tensor(row[f'{self.label}'], dtype=torch.long)


        return img, label
    
    def _load_image(self, photo_id):
        """Loads an image from the given directory and applies transformations."""
        img_path = f"{self.dir}/{photo_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img
    

class MultimodalDataset(Dataset):
    """Dataset for handling both image and text data, used for multimodal learning."""
    def __init__(self, df, dir, label, tokenizer=None, transform=None):
        self.df = df
        self.dir = dir
        self.label = label
        self.transform = transform if transform else transforms.ToTensor()
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return MultimodalDataset(self.df[idx], 
                                     self.dir, 
                                     self.label,
                                     tokenizer=self.tokenizer, 
                                     transform=self.transform)

        row = self.df.iloc[idx]

        photo_id = row["photo_id"]

        try:
            image = self._load_image(photo_id)
        except Exception as e:
            print(f"Warning: Could not load {photo_id} ({e})")
        
        text = row['summary']
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        label = torch.tensor(row[f'{self.label}'], dtype=torch.long)

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label
        }
    
    def _load_image(self, photo_id):
        img_path = f"{self.dir}/{photo_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img


def stratified_split_dataset(dataset, labels, train_size, val_size, random_state=None):
    """
    Splits a dataset into training, validation, and test sets using stratified sampling.
    
    Args:
        dataset: The dataset to be split.
        labels: Labels for stratified sampling.
        train_size: Proportion of data for training.
        val_size: Proportion of data for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Three subsets: train, validation, and test datasets.
    """
    test_size = 1.0 - train_size - val_size
    
    train_idx, temp_idx = train_test_split(
        range(len(dataset)),
        train_size=train_size,
        stratify=labels,
        random_state=random_state
    )
    
    temp_labels = labels[temp_idx]
    val_test_ratio = val_size / (val_size + test_size)
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_test_ratio,
        stratify=temp_labels,
        random_state=random_state
    )
    
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


# Data augmentation and preprocessing transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images randomly
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Color distortion
    RandAugment(num_ops=2, magnitude=9),  # Apply RandAugment
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Slight rotations & translations
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


val_transform = transforms.Compose([
     transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
