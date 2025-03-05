from rec_engine.database.yelp_index import reset_elasticsearch
import logging
import argparse
import asyncio

logger = logging.getLogger(__name__)

async def _main():
    parser = argparse.ArgumentParser(description="Clear the Elasticsearch index")
    parser.add_argument("--es_index_name", type=str, help="The name of the index to clear")
    parser.add_argument("--es_uri", type=str, help="The Elasticsearch URI", default="http://localhost:9200")
    parser.add_argument("--es_username", type=str, help="The Elasticsearch username", default="elastic")
    parser.add_argument("--es_password", type=str, help="The Elasticsearch password", default="yelp123")
    
    args = parser.parse_args()

    logger.info(f"Clearing index: {args.es_index_name}")
    await reset_elasticsearch(args.es_uri, args.es_username, args.es_password, args.es_index_name)
    logger.info(f"Successfully reset index: {args.es_index_name}")

if __name__ == "__main__":
    asyncio.run(_main()) 