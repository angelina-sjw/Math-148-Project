from pydantic import BaseModel
from datetime import datetime

class Review(BaseModel):
    rating: float
    comment: str
    date: datetime

class Restaurant(BaseModel):
    name: str
    address: str
    cuisine: str
    rating: float
    price_range: str
    reviews: list[Review]


