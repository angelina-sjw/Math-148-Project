from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Any, Optional

class Review(BaseModel):
    rating: float
    comment: str
    date: datetime

class RawRestaurant(BaseModel):
    business_id: str
    name: str
    address: str
    cuisine: str
    rating: float
    price_range: str
    # reviews: list[Review]
    photo_ids: list[str]
    captions: list[str]
    labels: list[str]

class Restaurant(BaseModel):
    business_id: str
    name: str
    address: str
    cuisine: str
    rating: float
    price_range: str
    profile_text_list: list[str]
    # reviews: list[Review]
    photo_ids: list[str]
    captions: list[str]
    labels: list[str]
    # These are properly annotated with Optional
    text_embeddings: Optional[List[List[float]]] = None
    photo_embeddings: Optional[List[List[float]]] = None
    
    model_config = {
        "extra": "allow"  # Allow extra fields not defined in the model
    }

class Result(BaseModel):
    restaurant: Restaurant
    score: float


schema1 = {
    "name": "get_keywords",
    "description": "Interpret predefined keywords from a user's query",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "cuisine": {
                "type": "string",
                "description": "The cuisine that user is looking for",
                "enum": ["Italian", "Chinese", "Mexican", "Indian", "Japanese"] 
            }
            # TODO: add more keywords
        },
        "required": ["cuisine"]
    }
}