from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import clip
from PIL import Image
import io
import numpy as np


app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

@app.post("/embed/text")
async def get_text_embedding(text: str):
    try:
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(device)
            embedding = model.encode_text(text_tokens)
            
            embedding = embedding.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding) #embedding is a numpy array
            
            return JSONResponse(content={"embedding": embedding.tolist()})
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing text: {str(e)}")

@app.post("/embed/image")
async def get_image_embedding(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data))
        image_input = preprocess(image_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            
            embedding = embedding.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return JSONResponse(content={"embedding": embedding.tolist()})
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

