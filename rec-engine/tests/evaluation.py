
import numpy as np
import os
import asyncio
import argparse
import collections

from rec_engine.core.agent import LLMClient
from rec_engine.database.engine import ESClient
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def evaluate_search_pipeline(es_uri="http://localhost:9200", es_username="elastic", 
                             es_password="yelp123", es_index_name="yelp_index",
                             clip_server_url="http://localhost:8000", openai_api_key=None):
    results = {
        "regular_search": {
            "text_queries": {},
            "image_queries": {},
            "aggregates": {}
        },
        "regular_search_with_keywords": {
            "text_queries": {},
            "aggregates": {}
        },
        "cross_modal_search": {
            "text_queries": {},
            "image_queries": {},
            "aggregates": {}
        },
        "cross_modal_search_with_keywords": {
            "text_queries": {},
            "aggregates": {}
        },
        "comparison": {}
    }
    
    try:
        es_client = ESClient(es_uri, es_username, es_password)
        llm_client = LLMClient(openai_api_key, clip_server_url)
        
        if not es_client.index_exists(es_index_name):
            logger.error(f"Index '{es_index_name}' does not exist in Elasticsearch!")
            return results
            
        logger.info(f"Connected to Elasticsearch, index '{es_index_name}' exists")
    except Exception as e:
        logger.error(f"Error connecting to services: {e}")
        return results
    
    text_queries = [
        "Italian restaurant with outdoor seating",
        "Cozy coffee shop with pastries",
        "Sushi restaurant with good reviews",
        "Mexican food truck",
        "Upscale steakhouse",
        "Vegan friendly cafe",
        "Sports bar with craft beer",
        "Family restaurant with kids menu",
        "Chinese takeout",
        "Breakfast diner"
    ]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_queries = [
        os.path.join(current_dir, "test_images/pizza_image.jpg"),
        os.path.join(current_dir, "test_images/coffee_shop_image.jpg"),
        os.path.join(current_dir, "test_images/sushi_plate_image.jpg"),
        os.path.join(current_dir, "test_images/taco_truck_image.jpg"),
        os.path.join(current_dir, "test_images/steak_dinner_image.jpg")
    ]
    
    test_images_dir = os.path.join(current_dir, "test_images")
    if not os.path.exists(test_images_dir):
        os.makedirs(test_images_dir)
        logger.info(f"Created test_images directory at {test_images_dir}. Please add test images before running.")
        
    missing_images = [img for img in image_queries if not os.path.exists(img)]
    if missing_images:
        logger.warning(f"The following test images are missing: {missing_images}")
        logger.warning("Please add these images before running the evaluation.")
    
    evaluation_results_dir = os.path.join(current_dir, "evaluation_results")
    if not os.path.exists(evaluation_results_dir):
        os.makedirs(evaluation_results_dir)
    
    logger.info("=== Evaluating Basic Regular Search ===")
    for query in text_queries:
        query_results = await es_client.search_restaurants(
            es_index_name, query, k=10, llm_client=llm_client, cross_modal=False
        )
        
        if query_results:
            results["regular_search"]["text_queries"][query] = {
                "num_results": len(query_results),
                "top_results": [
                    {
                        "name": r.restaurant.name,
                        "score": r.score,
                        "cuisine": r.restaurant.cuisine,
                        "rating": r.restaurant.rating,
                        "price_range": r.restaurant.price_range,
                        "address": r.restaurant.address
                    } 
                    for r in query_results[:5]
                ],
                "cuisines": [r.restaurant.cuisine for r in query_results],
                "ratings": [r.restaurant.rating for r in query_results],
                "price_ranges": [r.restaurant.price_range for r in query_results]
            }
    
    logger.info("=== Evaluating Advanced Regular Search ===")
    for query in text_queries:
        keywords = await llm_client.extract_keywords(query)
        
        query_results = await es_client.search_restaurants(
            es_index_name, query, k=10, llm_client=llm_client, 
            cross_modal=False, keywords=keywords
        )
        
        if query_results:
            results["regular_search_with_keywords"]["text_queries"][query] = {
                "num_results": len(query_results),
                "extracted_keywords": keywords,
                "top_results": [
                    {
                        "name": r.restaurant.name,
                        "score": r.score,
                        "cuisine": r.restaurant.cuisine,
                        "rating": r.restaurant.rating,
                        "price_range": r.restaurant.price_range,
                        "address": r.restaurant.address
                    } 
                    for r in query_results[:5]
                ],
                "cuisines": [r.restaurant.cuisine for r in query_results],
                "ratings": [r.restaurant.rating for r in query_results],
                "price_ranges": [r.restaurant.price_range for r in query_results]
            }
    
    for img_path in image_queries:
        if os.path.exists(img_path):
            query_results = await es_client.search_restaurants(
                es_index_name, img_path, k=10, llm_client=llm_client, cross_modal=False
            )
            
            if query_results:
                results["regular_search"]["image_queries"][img_path] = {
                    "num_results": len(query_results),
                    "top_results": [
                        {
                            "name": r.restaurant.name,
                            "score": r.score,
                            "cuisine": r.restaurant.cuisine,
                            "rating": r.restaurant.rating,
                            "price_range": r.restaurant.price_range,
                            "address": r.restaurant.address
                        } 
                        for r in query_results[:5]
                    ],
                    "cuisines": [r.restaurant.cuisine for r in query_results],
                    "ratings": [r.restaurant.rating for r in query_results],
                    "price_ranges": [r.restaurant.price_range for r in query_results]
                }
    
    logger.info("=== Evaluating Basic Cross-Modal Search ===")
    for query in text_queries:
        query_results = await es_client.search_restaurants(
            es_index_name, query, k=10, llm_client=llm_client, cross_modal=True
        )
        
        if query_results:
            results["cross_modal_search"]["text_queries"][query] = {
                "num_results": len(query_results),
                "top_results": [
                    {
                        "name": r.restaurant.name,
                        "score": r.score,
                        "cuisine": r.restaurant.cuisine,
                        "rating": r.restaurant.rating,
                        "price_range": r.restaurant.price_range,
                        "address": r.restaurant.address
                    } 
                    for r in query_results[:5]
                ],
                "cuisines": [r.restaurant.cuisine for r in query_results],
                "ratings": [r.restaurant.rating for r in query_results],
                "price_ranges": [r.restaurant.price_range for r in query_results]
            }
    
    logger.info("=== Evaluating Advanced Cross-Modal Search ===")
    for query in text_queries:
        keywords = await llm_client.extract_keywords(query)
        
        query_results = await es_client.search_restaurants(
            es_index_name, query, k=10, llm_client=llm_client, 
            cross_modal=True, keywords=keywords
        )
        
        if query_results:
            results["cross_modal_search_with_keywords"]["text_queries"][query] = {
                "num_results": len(query_results),
                "extracted_keywords": keywords,
                "top_results": [
                    {
                        "name": r.restaurant.name,
                        "score": r.score,
                        "cuisine": r.restaurant.cuisine,
                        "rating": r.restaurant.rating,
                        "price_range": r.restaurant.price_range,
                        "address": r.restaurant.address
                    } 
                    for r in query_results[:5]
                ],
                "cuisines": [r.restaurant.cuisine for r in query_results],
                "ratings": [r.restaurant.rating for r in query_results],
                "price_ranges": [r.restaurant.price_range for r in query_results]
            }
    
    for img_path in image_queries:
        if os.path.exists(img_path):
            query_results = await es_client.search_restaurants(
                es_index_name, img_path, k=10, llm_client=llm_client, cross_modal=True
            )
            
            if query_results:
                results["cross_modal_search"]["image_queries"][img_path] = {
                    "num_results": len(query_results),
                    "top_results": [
                        {
                            "name": r.restaurant.name,
                            "score": r.score,
                            "cuisine": r.restaurant.cuisine,
                            "rating": r.restaurant.rating,
                            "price_range": r.restaurant.price_range,
                            "address": r.restaurant.address
                        } 
                        for r in query_results[:5]
                    ],
                    "cuisines": [r.restaurant.cuisine for r in query_results],
                    "ratings": [r.restaurant.rating for r in query_results],
                    "price_ranges": [r.restaurant.price_range for r in query_results]
                }
    
    logger.info("=== Calculating Metrics and Comparisons ===")
    
    text_cuisine_diversity_regular = calculate_diversity([results["regular_search"]["text_queries"][q]["cuisines"] for q in results["regular_search"]["text_queries"]])
    text_cuisine_diversity_regular_keywords = calculate_diversity([results["regular_search_with_keywords"]["text_queries"][q]["cuisines"]  for q in results["regular_search_with_keywords"]["text_queries"]])
    text_cuisine_diversity_cross = calculate_diversity([results["cross_modal_search"]["text_queries"][q]["cuisines"] for q in results["cross_modal_search"]["text_queries"]])
    text_cuisine_diversity_cross_keywords = calculate_diversity([results["cross_modal_search_with_keywords"]["text_queries"][q]["cuisines"] for q in results["cross_modal_search_with_keywords"]["text_queries"]])
    avg_results_text_regular = np.mean([len(results["regular_search"]["text_queries"][q]["top_results"]) for q in results["regular_search"]["text_queries"]])
    avg_results_text_regular_keywords = np.mean([len(results["regular_search_with_keywords"]["text_queries"][q]["top_results"]) for q in results["regular_search_with_keywords"]["text_queries"]])
    avg_results_text_cross = np.mean([len(results["cross_modal_search"]["text_queries"][q]["top_results"]) for q in results["cross_modal_search"]["text_queries"]])
    avg_results_text_cross_keywords = np.mean([len(results["cross_modal_search_with_keywords"]["text_queries"][q]["top_results"]) for q in results["cross_modal_search_with_keywords"]["text_queries"]])
    image_cuisine_diversity_regular = calculate_diversity([results["regular_search"]["image_queries"][q]["cuisines"] for q in results["regular_search"]["image_queries"]])
    image_cuisine_diversity_cross = calculate_diversity([results["cross_modal_search"]["image_queries"][q]["cuisines"] for q in results["cross_modal_search"]["image_queries"]])
    avg_results_image_regular = np.mean([len(results["regular_search"]["image_queries"][q]["top_results"]) for q in results["regular_search"]["image_queries"]])
    avg_results_image_cross = np.mean([len(results["cross_modal_search"]["image_queries"][q]["top_results"]) for q in results["cross_modal_search"]["image_queries"]])
    
    results["regular_search"]["aggregates"] = {
        "text_cuisine_diversity": text_cuisine_diversity_regular,
        "image_cuisine_diversity": image_cuisine_diversity_regular,
        "avg_results_text": avg_results_text_regular,
        "avg_results_image": avg_results_image_regular
    }
    
    results["regular_search_with_keywords"]["aggregates"] = {
        "text_cuisine_diversity": text_cuisine_diversity_regular_keywords,
        "avg_results_text": avg_results_text_regular_keywords
    }
    
    results["cross_modal_search"]["aggregates"] = {
        "text_cuisine_diversity": text_cuisine_diversity_cross,
        "image_cuisine_diversity": image_cuisine_diversity_cross,
        "avg_results_text": avg_results_text_cross,
        "avg_results_image": avg_results_image_cross
    }
    
    results["cross_modal_search_with_keywords"]["aggregates"] = {
        "text_cuisine_diversity": text_cuisine_diversity_cross_keywords,
        "avg_results_text": avg_results_text_cross_keywords
    }
    
    results["comparison"] = {
        "text_cuisine_diversity_diff_cross_vs_regular": text_cuisine_diversity_cross - text_cuisine_diversity_regular,
        "image_cuisine_diversity_diff_cross_vs_regular": image_cuisine_diversity_cross - image_cuisine_diversity_regular,
        "text_results_count_diff_cross_vs_regular": avg_results_text_cross - avg_results_text_regular,
        "image_results_count_diff_cross_vs_regular": avg_results_image_cross - avg_results_image_regular,
        
        "text_cuisine_diversity_diff_keywords_regular": text_cuisine_diversity_regular_keywords - text_cuisine_diversity_regular,
        "text_cuisine_diversity_diff_keywords_cross": text_cuisine_diversity_cross_keywords - text_cuisine_diversity_cross,
        "text_results_count_diff_keywords_regular": avg_results_text_regular_keywords - avg_results_text_regular,
        "text_results_count_diff_keywords_cross": avg_results_text_cross_keywords - avg_results_text_cross,
        
        "text_cuisine_diversity_best_vs_basic": max(text_cuisine_diversity_regular_keywords, text_cuisine_diversity_cross, text_cuisine_diversity_cross_keywords) - text_cuisine_diversity_regular,
        "text_results_count_best_vs_basic": max(avg_results_text_regular_keywords, avg_results_text_cross, avg_results_text_cross_keywords) - avg_results_text_regular
    }
    
    logger.info("Generating evaluation report...")
    
    with open(os.path.join(evaluation_results_dir, 'evaluation_report.md'), 'w') as f:
        f.write('# Search Pipeline Evaluation Report\n\n')
        
        f.write('## Comparison of Four Search Approaches\n\n')
        f.write('This evaluation compares four search approaches for text queries:\n')
        f.write('1. **Basic Regular Search**: Text query → Text embeddings only\n')
        f.write('2. **Advanced Regular Search**: Text query → Extract keywords → Text embeddings only\n')
        f.write('3. **Basic Cross-Modal Search**: Text query → Text & image embeddings\n')
        f.write('4. **Advanced Cross-Modal Search**: Text query → Extract keywords → Text & image embeddings\n\n')
        
        f.write('For image queries, we compare two approaches:\n')
        f.write('1. **Regular Search**: Image query → Image embeddings only\n')
        f.write('2. **Cross-Modal Search**: Image query → Text & image embeddings\n\n')
        
        f.write('## Key Metrics\n\n')
        
        f.write('### Text Query Results\n\n')
        f.write('#### Cuisine Diversity (Shannon Index)\n\n')
        f.write('| Search Approach | Diversity Index | Difference from Basic |\n')
        f.write('|-----------------|----------------|-----------------------|\n')
        f.write(f'| Basic Regular | {text_cuisine_diversity_regular:.4f} | - |\n')
        f.write(f'| Advanced Regular (+ Keywords) | {text_cuisine_diversity_regular_keywords:.4f} | {text_cuisine_diversity_regular_keywords - text_cuisine_diversity_regular:.4f} |\n')
        f.write(f'| Basic Cross-Modal | {text_cuisine_diversity_cross:.4f} | {text_cuisine_diversity_cross - text_cuisine_diversity_regular:.4f} |\n')
        f.write(f'| Advanced Cross-Modal (+ Keywords) | {text_cuisine_diversity_cross_keywords:.4f} | {text_cuisine_diversity_cross_keywords - text_cuisine_diversity_regular:.4f} |\n\n')
        
        f.write('#### Average Number of Results\n\n')
        f.write('| Search Approach | Average Results | Difference from Basic |\n')
        f.write('|-----------------|----------------|-----------------------|\n')
        f.write(f'| Basic Regular | {avg_results_text_regular:.2f} | - |\n')
        f.write(f'| Advanced Regular (+ Keywords) | {avg_results_text_regular_keywords:.2f} | {avg_results_text_regular_keywords - avg_results_text_regular:.2f} |\n')
        f.write(f'| Basic Cross-Modal | {avg_results_text_cross:.2f} | {avg_results_text_cross - avg_results_text_regular:.2f} |\n')
        f.write(f'| Advanced Cross-Modal (+ Keywords) | {avg_results_text_cross_keywords:.2f} | {avg_results_text_cross_keywords - avg_results_text_regular:.2f} |\n\n')
        
        f.write('### Image Query Results\n\n')
        f.write('#### Cuisine Diversity (Shannon Index)\n\n')
        f.write('| Search Approach | Diversity Index | Difference |\n')
        f.write('|-----------------|----------------|------------|\n')
        f.write(f'| Regular Search | {image_cuisine_diversity_regular:.4f} | - |\n')
        f.write(f'| Cross-Modal Search | {image_cuisine_diversity_cross:.4f} | {image_cuisine_diversity_cross - image_cuisine_diversity_regular:.4f} |\n\n')
        
        f.write('#### Average Number of Results\n\n')
        f.write('| Search Approach | Average Results | Difference |\n')
        f.write('|-----------------|----------------|------------|\n')
        f.write(f'| Regular Search | {avg_results_image_regular:.2f} | - |\n')
        f.write(f'| Cross-Modal Search | {avg_results_image_cross:.2f} | {avg_results_image_cross - avg_results_image_regular:.2f} |\n\n')
        
        f.write('## Sample Query Results\n\n')
        
        for query in list(results["regular_search"]["text_queries"].keys())[:3]:
            if (query in results["regular_search_with_keywords"]["text_queries"] and
                query in results["cross_modal_search"]["text_queries"] and
                query in results["cross_modal_search_with_keywords"]["text_queries"]):
                
                f.write(f'### Text Query: "{query}"\n\n')
                
                f.write('#### Basic Regular Search Results:\n')
                for i, result in enumerate(results["regular_search"]["text_queries"][query]["top_results"][:3]):
                    f.write(f'{i+1}. {result["name"]} - Score: {result["score"]:.4f}\n')
                    f.write(f'   - Cuisine: {result["cuisine"]}\n')
                    f.write(f'   - Address: {result["address"] if "address" in result else "N/A"}\n')
                    f.write(f'   - Rating: {result["rating"]} stars\n')
                    f.write(f'   - Price Range: {result["price_range"]}\n\n')
                
                f.write('#### Advanced Regular Search (+ Keywords) Results:\n')
                f.write(f'Keywords extracted: {results["regular_search_with_keywords"]["text_queries"][query]["extracted_keywords"]}\n\n')
                for i, result in enumerate(results["regular_search_with_keywords"]["text_queries"][query]["top_results"][:3]):
                    f.write(f'{i+1}. {result["name"]} - Score: {result["score"]:.4f}\n')
                    f.write(f'   - Cuisine: {result["cuisine"]}\n')
                    f.write(f'   - Address: {result["address"] if "address" in result else "N/A"}\n')
                    f.write(f'   - Rating: {result["rating"]} stars\n')
                    f.write(f'   - Price Range: {result["price_range"]}\n\n')
                
                f.write('#### Basic Cross-Modal Search Results:\n')
                for i, result in enumerate(results["cross_modal_search"]["text_queries"][query]["top_results"][:3]):
                    f.write(f'{i+1}. {result["name"]} - Score: {result["score"]:.4f}\n')
                    f.write(f'   - Cuisine: {result["cuisine"]}\n')
                    f.write(f'   - Address: {result["address"] if "address" in result else "N/A"}\n')
                    f.write(f'   - Rating: {result["rating"]} stars\n')
                    f.write(f'   - Price Range: {result["price_range"]}\n\n')
                
                f.write('#### Advanced Cross-Modal Search (+ Keywords) Results:\n')
                f.write(f'Keywords extracted: {results["cross_modal_search_with_keywords"]["text_queries"][query]["extracted_keywords"]}\n\n')
                for i, result in enumerate(results["cross_modal_search_with_keywords"]["text_queries"][query]["top_results"][:3]):
                    f.write(f'{i+1}. {result["name"]} - Score: {result["score"]:.4f}\n')
                    f.write(f'   - Cuisine: {result["cuisine"]}\n')
                    f.write(f'   - Address: {result["address"] if "address" in result else "N/A"}\n')
                    f.write(f'   - Rating: {result["rating"]} stars\n')
                    f.write(f'   - Price Range: {result["price_range"]}\n\n')
                
                f.write('\n')
        
        for img_path in list(results["regular_search"]["image_queries"].keys())[:2]:
            if img_path in results["cross_modal_search"]["image_queries"]:
                f.write(f'### Image Query: "{os.path.basename(img_path)}"\n\n')
                
                f.write('#### Regular Search Results:\n')
                for i, result in enumerate(results["regular_search"]["image_queries"][img_path]["top_results"][:3]):
                    f.write(f'{i+1}. {result["name"]} - Score: {result["score"]:.4f}\n')
                    f.write(f'   - Cuisine: {result["cuisine"]}\n')
                    f.write(f'   - Address: {result["address"] if "address" in result else "N/A"}\n')
                    f.write(f'   - Rating: {result["rating"]} stars\n')
                    f.write(f'   - Price Range: {result["price_range"]}\n\n')
                
                f.write('#### Cross-Modal Search Results:\n')
                for i, result in enumerate(results["cross_modal_search"]["image_queries"][img_path]["top_results"][:3]):
                    f.write(f'{i+1}. {result["name"]} - Score: {result["score"]:.4f}\n')
                    f.write(f'   - Cuisine: {result["cuisine"]}\n')
                    f.write(f'   - Address: {result["address"] if "address" in result else "N/A"}\n')
                    f.write(f'   - Rating: {result["rating"]} stars\n')
                    f.write(f'   - Price Range: {result["price_range"]}\n\n')
                
                f.write('\n')
    
    logger.info(f"Evaluation complete! Results saved to the {evaluation_results_dir} directory")
    
    return results

def calculate_diversity(collections_list):
    if not collections_list or all(not collection for collection in collections_list):
        return 0.0
    
    all_items = []
    for collection in collections_list:
        all_items.extend(collection)
    
    if not all_items:
        return 0.0
    
    item_counts = collections.Counter(all_items)
    total = sum(item_counts.values())
    
    shannon_index = 0
    for count in item_counts.values():
        proportion = count / total
        shannon_index -= proportion * np.log(proportion)
    
    return shannon_index

async def run_evaluation():
    logger.info("Starting search pipeline evaluation...")
    
    parser = argparse.ArgumentParser(description="Evaluate search pipeline")
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API key", default="")
    parser.add_argument("--es_uri", type=str, help="Elasticsearch URI", default="http://localhost:9200")
    parser.add_argument("--es_username", type=str, help="Elasticsearch username", default="elastic")
    parser.add_argument("--es_password", type=str, help="Elasticsearch password", default="yelp123")
    parser.add_argument("--es_index_name", type=str, help="Elasticsearch index name", default="yelp_index")
    parser.add_argument("--clip_server_url", type=str, help="CLIP server URL", default="http://localhost:8000")
    args = parser.parse_args()
    
    if not args.openai_api_key:
        logger.warning("No OpenAI API key provided. Set --openai_api_key for best results.")
    
    results = await evaluate_search_pipeline(
        es_uri=args.es_uri,
        es_username=args.es_username, 
        es_password=args.es_password,
        es_index_name=args.es_index_name,
        clip_server_url=args.clip_server_url,
        openai_api_key=args.openai_api_key
    )
    
    return results

if __name__ == "__main__":
    asyncio.run(run_evaluation())