from rec_engine.core.agent import LLMClient
from rec_engine.database.engine import ESClient
import logging
import argparse
import os
import asyncio
import sys

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   stream=sys.stdout) 
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="The user's text query or path to an image")
    parser.add_argument("--openai_api_key", type=str, help="The OpenAI API key")
    parser.add_argument("--advanced_mode", action="store_true", help="Whether to run in advanced mode")
    parser.add_argument("--es_index_name", type=str, help="The name of the index", default="yelp_index")
    parser.add_argument("--clip_server_url", type=str, help="The CLIP server URL", default="http://localhost:8000")
    parser.add_argument("--es_uri", type=str, help="The Elasticsearch URI", default="http://localhost:9200")
    parser.add_argument("--es_username", type=str, help="The Elasticsearch username", default="elastic")
    parser.add_argument("--es_password", type=str, help="The Elasticsearch password", default="yelp123")
    parser.add_argument("--k", type=int, help="Number of results to return", default=5)
    args = parser.parse_args()

    query = args.query
    print(f"Searching for: {query}") 
    
    try:
        es_client = ESClient(args.es_uri, args.es_username, args.es_password)
        llm_client = LLMClient(args.openai_api_key, args.clip_server_url)
        
        if not es_client.index_exists(args.es_index_name):
            print(f"Error: Index '{args.es_index_name}' does not exist in Elasticsearch!")
            return
            
        print(f"Connected to Elasticsearch, index '{args.es_index_name}' exists")
        
        is_image_query = os.path.exists(query) and os.path.isfile(query)
        print(f"Query type: {'image' if is_image_query else 'text'}")
        
        if args.advanced_mode and not is_image_query:
            print("Using advanced mode with keyword extraction")
            keywords = await llm_client.extract_keywords(query)
            print(f"Extracted keywords: {keywords}")
            results = await es_client.search_restaurants(args.es_index_name, query, args.k, llm_client, keywords)
        else:
            print("Using standard search")
            results = await es_client.search_restaurants(args.es_index_name, query, args.k, llm_client)
        
        print(f"Search complete, found {len(results)} results")
        
        if not results:
            print("No matching restaurants found. Try a different query.")
            return
            
        print("\n=== RESULTS ===")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.restaurant.name} - Score: {result.score:.2f}")
            print(f"   Address: {result.restaurant.address}")
            print(f"   Cuisine: {result.restaurant.cuisine}")
            print(f"   Rating: {result.restaurant.rating} stars")
            print(f"   Price Range: {result.restaurant.price_range}")
            print()
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())