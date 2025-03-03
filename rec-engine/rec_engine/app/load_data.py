from rec_engine.database.yelp_index import intialize_yelp_index
import logging
import argparse
import asyncio

logger = logging.getLogger(__name__)

async def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_api_key", type=str, help="The OpenAI API key")
    parser.add_argument("--test_mode", action="store_true", help="Whether to run in test mode")
    parser.add_argument("--clip_server_url", type=str, help="The CLIP server URL", default="http://localhost:8000")
    parser.add_argument("--es_uri", type=str, help="The Elasticsearch URI", default="http://localhost:9200")
    parser.add_argument("--es_username", type=str, help="The Elasticsearch username", default="elastic")
    parser.add_argument("--es_password", type=str, help="The Elasticsearch password", default="yelp123")
    parser.add_argument("--data_dir", type=str, help="The directory of data", default="rec_engine/data")
    parser.add_argument("--es_index_name", type=str, help="The name of the index", default="yelp_index")
    args = parser.parse_args()

    await intialize_yelp_index(
        data_dir=args.data_dir,
        openai_api_key=args.openai_api_key,
        clip_server_url=args.clip_server_url,
        es_uri=args.es_uri,
        es_username=args.es_username,
        es_password=args.es_password,
        es_index_name=args.es_index_name,
        test_mode=args.test_mode
    )

if __name__ == "__main__":
    asyncio.run(_main())
