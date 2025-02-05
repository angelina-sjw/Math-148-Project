from core import query_agent
import logging
import argparse

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="The user's query")
    parser.add_argument("--openai_api_key", type=str, help="The OpenAI API key")
    parser.add_argument("--es_uri", type=str, help="The Elasticsearch URI")
    parser.add_argument("--es_username", type=str, help="The Elasticsearch username")
    parser.add_argument("--es_password", type=str, help="The Elasticsearch password")
    args = parser.parse_args()

    query = args.query
    keywords = query_agent.extract_keywords(query)
    logger.info(f"Extracted keywords: {keywords}")


if __name__ == "__main__":
    main()