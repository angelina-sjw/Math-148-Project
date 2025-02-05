import json
from typing import Dict
from ..database.engine import ESClient
from ..data_types import RawRestaurant, Restaurant
from ..core.agent import LLMClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _generate_restaurant_profile(restaurant: Restaurant) -> Dict:
    price_descriptions = {
        "1": "budget-friendly",
        "2": "moderately priced",
        "3": "upscale",
        "4": "fine dining"
    }
    price_text = price_descriptions.get(restaurant.price_range,"")

    cuisines = restaurant.cuisine.split(",")
    if len(cuisines) > 1:
        cuisine_text = " and ".join(cuisines)
    else:
        cuisine_text = cuisines[0]

    profile = f"{restaurant.name} is a {price_text} {cuisine_text} restaurant located at {restaurant.address}. The restaurant have highlighted the following features: {restaurant.caption}"

    if restaurant.rating:
        profile += f"With a rating of {restaurant.rating} stars, "
        if restaurant.rating >= 4.5:
            profile += "it's one of the highest-rated establishments in the area. "
        elif restaurant.rating >= 4.0:
            profile += "it's very well-reviewed by customers. "
        elif restaurant.rating >= 3.5:
            profile += "it receives generally positive reviews. "
        else:
            profile += "it provides a casual dining experience. "

    if hasattr(restaurant, 'reviews') and restaurant.reviews:
        recent_reviews = sorted(restaurant.reviews, key=lambda x: x.date, reverse=True)[:3]
        if recent_reviews:
            profile += "Recent customer feedback includes: "
            for review in recent_reviews:
                profile += f'"{review.comment}" '

    return profile.strip()


async def intialize_yelp_index(data_dir: str, openai_api_key: str, es_uri: str, es_username: str, es_password: str, es_index_name: str = "yelp_index", batch_size: int = 1000):
    llm_client = LLMClient(openai_api_key)
    es_client = ESClient(es_uri, es_username, es_password)
    es_client.create_index(es_index_name) if not es_client.index_exists(es_index_name) else None
    photos = {}
    try:
        with open(f"{data_dir}/photos.json", "r") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    business_id = item["business_id"]
                    photos[business_id] = photos.get(business_id, []) + [item]
                except json.JSONDecodeError:
                    continue
                except KeyError:
                    continue

    except Exception as e:
        logger.error(f"Error loading photos.json: {e}")
        return False
    try:    
        with open(f"{data_dir}/yelp_academic_dataset_business.json", "r") as f:
            businesses = []
            for line in f: 
                business = json.loads(line)
                if business["business_id"] in photos:
                    for photo_profile in photos[business["business_id"]]:
                        restaurant = RawRestaurant(
                            business_id=business.get("business_id", ""),
                            name=business.get("name", ""),
                            address=business.get("address", ""),
                            cuisine=business.get("categories", ""),
                            rating=business.get("stars", ""),
                            price_range=business.get("RestaurantPriceRange2", ""),
                            photo_id=photo_profile.get("photo_id", ""),
                            caption=photo_profile.get("caption", ""),
                            label=photo_profile.get("label", "")
                        )

                        profile_text = _generate_restaurant_profile(restaurant)

                        logger.info(f"Generating text embedding for {restaurant.name}")
                        text_embedding = await llm_client.get_embedding(profile_text, "text")
                        logger.info(f"Generating photo embedding for {restaurant.name}")
                        with open(f"data/photos/{photo_profile['photo_id']}.jpg", "rb") as img_file:
                            photo_data = img_file.read()
                        photo_embedding = await llm_client.get_embedding(photo_data, "image")
                        logger.info(f"Generated embeddings for {restaurant.name}")

                        restaurant = Restaurant(
                            **restaurant.model_dump(),
                            profile_text=profile_text,
                            text_embedding=text_embedding,
                            photo_embedding=photo_embedding
                        )
                        businesses.append(restaurant)

                        if len(businesses) >= batch_size:
                            await es_client.add_restaurant_bulk(es_index_name, businesses)
                            businesses = []
        return True
    except Exception as e:
        logger.error(f"Error loading yelp_academic_dataset_business.json: {e}")
        return False

    
    
