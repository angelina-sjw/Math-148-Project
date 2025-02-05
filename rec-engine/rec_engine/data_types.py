from pydantic import BaseModel
from datetime import datetime

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
    photo_id: str
    caption: str
    label: str

class Restaurant(BaseModel):
    business_id: str
    name: str
    address: str
    cuisine: str
    rating: float
    price_range: str
    profile_text: str
    # reviews: list[Review]
    photo_id: str
    caption: str
    label: str
    text_embedding: list[float]
    photo_embedding: list[float]


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