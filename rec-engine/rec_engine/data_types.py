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
    text_embeddings: Optional[List[List[float]]] = None
    photo_embeddings: Optional[List[List[float]]] = None
    
    model_config = {
        "extra": "allow"
    }

class Result(BaseModel):
    restaurant: Restaurant
    score: float


schema1 = {
    "name": "get_keywords",
    "description": "Interpret predefined keywords from a user's query",
    "schema": {
        "type": "object",
        "properties": {
            "cuisine": {
                "type": "string",
                "description": "The cuisine that user is looking for",
                "enum": ["American", "Italian", "Chinese", "Mexican", "Indian", "Japanese", "Korean", "Thai", "Vietnamese", "French", "German", "Spanish", "Mediterranean", "Greek", "Turkish", "Brazilian"] 
            },
            "price_range": {
                "type": "string",
                "description": "The price range the user is looking for (1-4, where 1 is least expensive, 4 is most expensive)",
                "enum": ["1", "2", "3", "4"]
            },
            "rating": {
                "type": "number",
                "description": "The minimum rating (1-5 stars) the user is looking for",
                "enum": [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
            },
        },
        "additionalProperties": False,
        "required": ["cuisine", "price_range", "rating"]
    },
    "strict": True
}