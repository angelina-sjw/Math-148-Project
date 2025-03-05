import torch
import clip
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting CLIP model test")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("__02nEL2xViYvZihvV4_hw.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

logger.info("CLIP model is working with CUDA!" if torch.cuda.is_available() else "CUDA not detected.")
