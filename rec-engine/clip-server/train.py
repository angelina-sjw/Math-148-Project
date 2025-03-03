import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
from typing import Dict, List, Tuple
import os
from tqdm import tqdm
import sys

logger = logging.getLogger(__name__)

def _get_image_caption_pairs(photos_dataset_dir: str) -> List[Tuple[str, str]]:
    image_caption_pairs = []
    try:
        with open(f"{photos_dataset_dir}/photos.json", "r") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if item["caption"] is None or item["caption"] == "":
                        continue
                    photo_id = item["photo_id"]
                    caption = item["caption"]
                    image_caption_pairs.append((photo_id, caption))
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON: {line}")
                    continue
                except KeyError:
                    logger.error(f"Key error: {line}")
                    continue
    except Exception as e:
        logger.error(f"Error loading photos.json: {e}")
        return False
    return image_caption_pairs

class YelpImageTextDataset(Dataset):
    def __init__(self, photos_dataset_dir, transform=None, max_text_per_image=1):
        self.photos_dir = os.path.join(photos_dataset_dir, "photos")  # Look for images in the photos subdirectory
        self.transform = transform
        self.max_text_per_image = max_text_per_image
        
        # Load photo metadata from JSON file
        json_path = os.path.join(photos_dataset_dir, "photos.json")
        self.valid_photos = []
        
        # Check if directory and JSON file exist
        if not os.path.exists(photos_dataset_dir):
            logger.error(f"Photos dataset directory does not exist: {photos_dataset_dir}")
            raise ValueError(f"Photos dataset directory not found: {photos_dataset_dir}")
            
        if not os.path.exists(self.photos_dir):
            logger.error(f"Photos directory does not exist: {self.photos_dir}")
            raise ValueError(f"Photos directory not found: {self.photos_dir}")
            
        if not os.path.exists(json_path):
            logger.error(f"photos.json file not found at: {json_path}")
            raise ValueError(f"photos.json file not found at: {json_path}")
            
        logger.info(f"Reading photos data from: {json_path}")
        logger.info(f"Looking for images in: {self.photos_dir}")
        
        # Read file as JSONL (JSON Lines) where each line is a separate JSON object
        try:
            photo_count = 0
            with open(json_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    try:
                        if line_idx < 5:  # Print first few lines for debugging
                            logger.info(f"Sample line {line_idx}: {line[:100]}...")
                            
                        photo_count += 1
                        photo_data = json.loads(line.strip())
                        photo_id = photo_data.get('photo_id')
                        caption = photo_data.get('caption', '')
                        
                        if not photo_id:
                            logger.debug(f"Missing photo_id in line {line_idx}")
                            continue
                            
                        if not caption or not isinstance(caption, str) or not caption.strip():
                            logger.debug(f"Missing/invalid caption for photo_id: {photo_id}")
                            continue
                        
                        # Check if the image file exists
                        img_path = os.path.join(self.photos_dir, photo_id + '.jpg')
                        if not os.path.exists(img_path):
                            # Try without the .jpg extension
                            img_path = os.path.join(self.photos_dir, photo_id)
                            if not os.path.exists(img_path):
                                logger.debug(f"Image file not found: {img_path}")
                                continue
                            
                        self.valid_photos.append((photo_id, caption))
                        
                        if line_idx % 1000 == 0 and len(self.valid_photos) > 0:
                            logger.info(f"Processed {line_idx} lines, found {len(self.valid_photos)} valid photos")
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON line {line_idx}: {line[:100]}...")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing JSON line {line_idx}: {e}")
                        continue
            
            # List some images in the directory for debugging
            try:
                some_files = os.listdir(self.photos_dir)[:10]
                logger.info(f"Some files in photos directory: {some_files}")
            except Exception as e:
                logger.warning(f"Could not list files in photos directory: {e}")
                        
            logger.info(f"Processed {photo_count} photos, found {len(self.valid_photos)} valid photo-caption pairs")
            
            if len(self.valid_photos) == 0:
                logger.error("No valid photo-caption pairs found. Please check your dataset structure.")
                logger.error("Make sure images are in the 'photos' subdirectory of your data path.")
                raise ValueError("No valid photo-caption pairs found in the dataset")
                
        except Exception as e:
            logger.error(f"Error loading photos.json: {e}")
            raise e
    
    def __len__(self):
        return len(self.valid_photos)
    
    def __getitem__(self, idx):
        try:
            # Get the data for the requested index
            item = self.valid_photos[idx]
            
            # Check if item is a dictionary (expected format)
            if isinstance(item, tuple):
                photo_id, caption = item
            else:
                # If item is a tuple or some other format, try to extract properly
                logger.error(f"Error in __getitem__ for idx {idx}: unexpected item format {type(item)}")
                return None, None
            
            # Construct image path
            img_path = os.path.join(self.photos_dir, f"{photo_id}.jpg")
            
            # Load and transform the image
            try:
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    return image, caption
                else:
                    logger.error(f"Image not found: {img_path}")
                    return None, None
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                return None, None
            
        except Exception as e:
            logger.error(f"Error in __getitem__ for idx {idx}: {e}")
            return None, None

def train_clip(
    photos_dataset_dir: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # Verify the directory exists
    logger.info(f"Reading dataset from: {photos_dataset_dir}")
    
    # Load the model, image transform, and tokenizer
    model, _, image_preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    
    model = model.to(device)
    
    # Create dataset with better error handling
    try:
        dataset = YelpImageTextDataset(photos_dataset_dir, transform=image_preprocess)
        logger.info(f"Dataset loaded successfully with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        valid_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                # Check if batch_data is valid
                if batch_data[0] is None or batch_data[1] is None:
                    logger.warning(f"Skipping invalid batch {batch_idx}")
                    continue
                
                # Unpack the batch - make sure we're getting exactly 2 items
                images, texts = batch_data
                
                # Check that we have data
                if len(images) == 0 or len(texts) == 0:
                    logger.warning(f"Empty batch: images={len(images)}, texts={len(texts)}")
                    continue
                
                # If we have more texts than images, trim texts
                if len(texts) > len(images):
                    texts = texts[:len(images)]
                # If we have more images than texts, this shouldn't happen with our dataset
                elif len(images) > len(texts):
                    logger.warning(f"More images ({len(images)}) than texts ({len(texts)}) - this shouldn't happen")
                    continue
                
                # Move data to device
                images = images.to(device)
                text_tokens = tokenizer(texts).to(device)
                
                # Forward pass
                logits_per_image, logits_per_text = model(images, text_tokens)
                
                # Compute contrastive loss in both directions
                ground_truth = torch.arange(len(images)).to(device)
                loss = (
                    torch.nn.functional.cross_entropy(logits_per_image, ground_truth) +
                    torch.nn.functional.cross_entropy(logits_per_text, ground_truth)
                ) / 2
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update stats
                total_loss += loss.item()
                valid_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({"Loss": f"{total_loss / valid_batches:.2f}"})
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue

        if valid_batches == 0:
            logger.warning(f"No valid batches in epoch {epoch + 1}")
            continue
            
        avg_loss = total_loss / valid_batches
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Valid Batches: {valid_batches}/{len(dataloader)}")

        # Save checkpoint
        checkpoint_path = f"clip_checkpoint_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    return model

def collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    
    if len(batch) == 0:
        return None, None
    
    # Separate images and texts
    images, texts = zip(*batch)
    
    # Stack images into a batch
    try:
        images = torch.stack(images)
        # Return exactly two values: images tensor and list of texts
        return images, list(texts)
    except Exception as e:
        logger.error(f"Error stacking images in collate_fn: {e}")
        return None, None

if __name__ == "__main__":
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Train CLIP model on Yelp photos dataset")
    parser.add_argument("--photos_dir", type=str, required=True, help="Path to the directory containing photos and photos.json")
    parser.add_argument("--model_name", type=str, default="ViT-B-32", help="CLIP model architecture")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained weights to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    logger.info(f"Starting training with arguments: {args}")
    
    # Check if the directory exists
    if not os.path.exists(args.photos_dir):
        logger.error(f"Photos directory does not exist: {args.photos_dir}")
        sys.exit(1)
        
    photos_json = os.path.join(args.photos_dir, "photos.json")
    if not os.path.exists(photos_json):
        logger.error(f"photos.json not found at: {photos_json}")
        sys.exit(1)
    
    # Set CUDA options for debugging
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    trained_model = train_clip(
        photos_dataset_dir=args.photos_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )

