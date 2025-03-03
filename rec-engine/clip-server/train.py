import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
from typing import Dict, List, Tuple
import os
from tqdm import tqdm

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
    def __init__(self, photo_dataset_dir: str, transform=None):
        self.photo_dataset_dir = photo_dataset_dir
        self.transform = transform
        self.image_caption_pairs = _get_image_caption_pairs(photo_dataset_dir)
        
    def __len__(self):
        return len(self.image_caption_pairs)
    
    def __getitem__(self, idx):
        photo_id, caption = self.image_caption_pairs[idx]
        image_path = os.path.join(f"{self.photo_dataset_dir}/photos/", f"{photo_id}.jpg")
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, caption
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

def train_clip(
    photos_dataset_dir: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    dataset = YelpImageTextDataset(photos_dataset_dir, transform=preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, (images, texts) in enumerate(progress_bar):
            if images is None or texts is None:
                continue
                
            images = images.to(device)
            
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            
            # Normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            labels = torch.arange(len(images), device=device)
            loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # applies previous gradients to the model parameters to minimize loss

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": total_loss / (batch_idx + 1)})

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

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
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    
    images, texts = zip(*batch)
    images = torch.stack(images)
    return images, texts

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CLIP model on Yelp dataset")
    parser.add_argument("--photos_dir", type=str, required=True, help="Directory containing photos and photos.json")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--model_name", type=str, default="ViT-B-32", help="CLIP model architecture")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained model weights")
    
    args = parser.parse_args()
    
    trained_model = train_clip(
        photos_dataset_dir=args.photos_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )

