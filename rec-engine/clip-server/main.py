from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import torch
import clip
from PIL import Image
import io
import numpy as np
from pydantic import BaseModel
import base64

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class TextRequest(BaseModel):
    text: str

class ImageRequest(BaseModel):
    image_data: bytes

@app.post("/embed/text")
async def get_text_embedding(request: TextRequest):
    try:
        with torch.no_grad():
            text_tokens = clip.tokenize([request.text]).to(device)
            embedding = model.encode_text(text_tokens)
            
            embedding = embedding.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return JSONResponse(content={"embedding": embedding.tolist()})
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing text: {str(e)}")

@app.post("/embed/image")
async def get_image_embedding(request: ImageRequest):
    try:
        image_bytes = base64.b64decode(request.image_data)
        image_pil = Image.open(io.BytesIO(image_bytes))
        image_input = preprocess(image_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            
            embedding = embedding.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return JSONResponse(content={"embedding": embedding.tolist()})
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

