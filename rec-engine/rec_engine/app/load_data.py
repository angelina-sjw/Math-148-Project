from rec_engine.database.yelp_index import intialize_yelp_index
import logging
import argparse
import asyncio

logger = logging.getLogger(__name__)

async def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_api_key", type=str, help="The OpenAI API key")
    parser.add_argument("--es_uri", type=str, help="The Elasticsearch URI")
    parser.add_argument("--es_username", type=str, help="The Elasticsearch username")
    parser.add_argument("--es_password", type=str, help="The Elasticsearch password")
    parser.add_argument("--data_dir", type=str, help="The directory of data")
    args = parser.parse_args()

    await intialize_yelp_index(args.data_dir, args.openai_api_key, args.es_uri, args.es_username, args.es_password)

if __name__ == "__main__":
    asyncio.run(_main())
