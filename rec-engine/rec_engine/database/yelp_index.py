import json
from typing import Dict, List
from ..database.engine import ESClient
from ..data_types import RawRestaurant, Restaurant
from ..core.agent import LLMClient
import logging
import asyncio
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _generate_restaurant_profile(restaurant: RawRestaurant) -> List[str]:
    #TODO: change the profile generation to one sentence description because CLIP only supports 77 tokens
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

    sections = []

    basic_info = f"{restaurant.name} is a {price_text} {cuisine_text} restaurant located at {restaurant.address}."
    sections.append(basic_info)

    if hasattr(restaurant, 'caption') and restaurant.caption:
        features = f"Key features: {restaurant.caption}"
        sections.append(features)
    
    if hasattr(restaurant, 'rating') and restaurant.rating:
        rating_text = f"With a rating of {restaurant.rating} stars, "
        if restaurant.rating >= 4.5:
            rating_text += "it's one of the highest-rated establishments in the area. "
        elif restaurant.rating >= 4.0:
            rating_text += "it's very well-reviewed by customers. "
        elif restaurant.rating >= 3.5:
            rating_text += "it receives generally positive reviews. "
        else:
            rating_text += "it provides a casual dining experience. "
        sections.append(rating_text)
    
    if hasattr(restaurant, 'reviews') and restaurant.reviews:
        review_text = "Recent customer feedback includes: "
        recent_reviews = sorted(restaurant.reviews, key=lambda x: x.date, reverse=True)[:3]
        if recent_reviews:
            for review in recent_reviews:
                review_text += f'"{review.comment}" '
        sections.append(review_text)


    return sections

async def reset_elasticsearch(es_uri: str, es_username: str, es_password: str, es_index_name: str = "yelp_index"):
    """Completely reset Elasticsearch and index"""
    logger.info("Starting complete Elasticsearch reset")
    
    es_client = ESClient(es_uri, es_username, es_password)
    
    try:
        logger.info(f"Force deleting index {es_index_name}")
        es_client.es_client.indices.delete(index=es_index_name, ignore=[404])
    except Exception as e:
        logger.warning(f"Error during index deletion: {e}")
    
    await asyncio.sleep(2)
    
    try:
        logger.info("Clearing any pending tasks")
        es_client.es_client.tasks.cancel(actions="*", nodes="_all")
    except Exception as e:
        logger.warning(f"Error cancelling tasks: {e}")
    
    try:
        logger.info("Flushing all indices")
        es_client.es_client.indices.flush(index="_all", force=True)
    except Exception as e:
        logger.warning(f"Error flushing indices: {e}")
    
    try:
        logger.info("Optimizing remaining indices")
        es_client.es_client.indices.forcemerge(index="_all", max_num_segments=1)
    except Exception as e:
        logger.warning(f"Error optimizing indices: {e}")
    
    logger.info("Elasticsearch reset complete. Please restart Elasticsearch service for best results.")
    


async def intialize_yelp_index(data_dir: str, openai_api_key: str, clip_server_url: str, es_uri: str, es_username: str, es_password: str, es_index_name: str = "yelp_index", batch_size: int = 500, test_mode: bool = False):
    llm_client = LLMClient(openai_api_key, clip_server_url)
    es_client = ESClient(es_uri, es_username, es_password)
    es_client.create_index(es_index_name) if not es_client.index_exists(es_index_name) else None
    photos = {}

    # for testing
    photos_processed = 0
    business_processed = 0
    businesses_with_photos = 0

    try:
        with open(f"{data_dir}/photos.json", "r") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    business_id = item["business_id"]
                    photos[business_id] = photos.get(business_id, []) + [item]
                    photos_processed += 1
                    if test_mode and photos_processed >= 2000:
                        break
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
                try:  # Add a try/except for each business
                    business = json.loads(line)
                    business_processed += 1
                    
                    if test_mode and business_processed >= 2000:
                        logger.info(f"Test mode: Processed {business_processed} businesses")
                        break
                        
                    # Skip businesses without a business_id
                    if not business.get("business_id"):
                        logger.warning(f"Skipping business with no ID: {business.get('name', 'Unknown')}")
                        continue
                        
                    # Check if the business has photos
                    if business["business_id"] in photos:
                        businesses_with_photos += 1
                        
                        photo_embeddings = []
                        captions = []
                        photo_ids = []
                        labels = []

                        for photo_profile in photos[business["business_id"]]:
                            try:  # Add try/except around photo processing
                                # Check if photo exists
                                photo_path = f"{data_dir}/photos/{photo_profile['photo_id']}.jpg"
                                if not os.path.exists(photo_path):
                                    logger.warning(f"Photo {photo_profile['photo_id']} not found, skipping")
                                    continue
                                    
                                with open(photo_path, "rb") as img_file:
                                    photo_data = img_file.read()
                                photo_embedding = await llm_client.get_embedding(photo_data, "image")
                                photo_embeddings.append(photo_embedding)
                                captions.append(photo_profile.get("caption", ""))
                                photo_ids.append(photo_profile.get("photo_id", ""))
                            except Exception as e:
                                logger.error(f"Error processing photo: {e}")
                                continue

                        # Skip if no photo embeddings were created
                        if not photo_embeddings:
                            logger.warning(f"No valid photos for {business.get('name', 'Unknown')}, skipping")
                            continue

                        # Extract price range from attributes field safely
                        price_range = ""
                        if business.get("attributes"):
                            if isinstance(business["attributes"], dict):
                                price_range = business["attributes"].get("RestaurantsPriceRange2", "")
                            elif isinstance(business["attributes"], str):
                                # Try to parse if it's a string
                                try:
                                    attrs = json.loads(business["attributes"].replace("'", '"'))
                                    if isinstance(attrs, dict):
                                        price_range = attrs.get("RestaurantsPriceRange2", "")
                                except:
                                    pass

                        restaurant = RawRestaurant(
                            business_id=business.get("business_id", ""),
                            name=business.get("name", ""),
                            address=business.get("address", ""),
                            cuisine=business.get("categories", ""),
                            rating=business.get("stars", 0.0),
                            price_range=price_range,
                            photo_ids=photo_ids,
                            captions=captions,
                            labels=labels
                        )

                        profile_text_list = _generate_restaurant_profile(restaurant)

                        text_embeddings = []
                        for section in profile_text_list:
                            text_embedding = await llm_client.get_embedding(section, "text")
                            text_embeddings.append(text_embedding)
                        
                        # Create a base restaurant
                        restaurant_data = restaurant.model_dump()
                        restaurant_data["profile_text_list"] = profile_text_list
                        
                        # Add flattened embeddings with numeric indices
                        for i, emb in enumerate(photo_embeddings):
                            restaurant_data[f"photo_embedding_{i}"] = emb
                        for i, emb in enumerate(text_embeddings):
                            restaurant_data[f"text_embedding_{i}"] = emb
                        
                        # Create restaurant with flattened fields
                        restaurant = Restaurant(**restaurant_data)
                        businesses.append(restaurant)

                        if len(businesses) >= batch_size:
                            logger.info(f"Adding {len(businesses)} restaurants to database")
                            es_client.add_restaurant_bulk(es_index_name, businesses)
                            businesses = []
                except Exception as e:
                    logger.error(f"Error processing business: {str(e)}")
                    # Continue with next business
                    continue
            
            # Add any remaining businesses
            if businesses:
                logger.info(f"Adding final {len(businesses)} restaurants to database")
                es_client.add_restaurant_bulk(es_index_name, businesses)
            
            # Force a refresh to make documents searchable immediately
            logger.info("Refreshing index to make documents searchable")
            es_client.es_client.indices.refresh(index=es_index_name)
            
            logger.info(f"Total businesses with photos: {businesses_with_photos}")
            
        return True
    except Exception as e:
        logger.error(f"Error loading yelp_academic_dataset_business.json: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    
async def clear_yelp_index(es_uri: str, es_username: str, es_password: str, es_index_name: str):
    es_client = ESClient(es_uri, es_username, es_password)
    es_client.delete_index(es_index_name)
    es_client.create_index(es_index_name)
