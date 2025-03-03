import asyncio
import argparse
import logging
from elasticsearch import AsyncElasticsearch
from tabulate import tabulate
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def inspect_index(es_uri: str, es_username: str, es_password: str, es_index_name: str = "yelp_index"):
    """Get basic information about the Elasticsearch index"""
    logger.info(f"Inspecting index: {es_index_name}")
    
    # Create Elasticsearch client
    es = AsyncElasticsearch(
        es_uri,
        basic_auth=(es_username, es_password),
        request_timeout=30
    )
    
    # Check if index exists
    if not await es.indices.exists(index=es_index_name):
        logger.error(f"Index {es_index_name} does not exist")
        await es.close()
        return
    
    # Get index stats
    logger.info(f"Getting stats for index {es_index_name}")
    stats = await es.indices.stats(index=es_index_name)
    total_docs = stats["indices"][es_index_name]["total"]["docs"]["count"]
    index_size = stats["indices"][es_index_name]["total"]["store"]["size_in_bytes"] / 1024 / 1024  # Convert to MB
    
    # Get index mapping
    logger.info(f"Getting mapping for index {es_index_name}")
    mapping = await es.indices.get_mapping(index=es_index_name)
    fields = list(mapping[es_index_name]["mappings"]["properties"].keys())
    
    # Get a sample of documents
    logger.info(f"Getting sample documents from {es_index_name}")
    sample_query = {
        "size": 5,
        "_source": True,  # Get all fields
        "query": {
            "match_all": {}
        }
    }
    sample_results = await es.search(index=es_index_name, body=sample_query)
    
    # Display results
    print("\n" + "="*50)
    print(f"INDEX INFORMATION: {es_index_name}")
    print("="*50)
    print(f"Total documents: {total_docs}")
    print(f"Index size: {index_size:.2f} MB")
    print(f"Fields: {', '.join(fields)}")
    
    # Display sample documents in a table
    if sample_results["hits"]["total"]["value"] > 0:
        print("\nSAMPLE RESTAURANTS:")
        table_data = []
        headers = ["Business ID", "Name", "Cuisine", "Rating", "Price"]
        
        for hit in sample_results["hits"]["hits"]:
            source = hit["_source"]
            table_data.append([
                source.get("business_id", "N/A")[:10] + "...",
                source.get("name", "N/A"),
                source.get("cuisine", "N/A")[:20] + "..." if len(source.get("cuisine", "")) > 20 else source.get("cuisine", "N/A"),
                source.get("rating", "N/A"),
                source.get("price_range", "N/A")
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print("\nNo documents found in the index.")
    
    # Get some aggregate statistics
    logger.info("Getting aggregate statistics")
    aggs_query = {
        "size": 0,
        "aggs": {
            "avg_rating": {"avg": {"field": "rating"}},
            "price_ranges": {"terms": {"field": "price_range"}},
            "cuisines": {"terms": {"field": "cuisine.keyword", "size": 10}}
        }
    }
    
    try:
        aggs_results = await es.search(index=es_index_name, body=aggs_query)
        
        print("\nAGGREGATE STATISTICS:")
        if "avg_rating" in aggs_results["aggregations"]:
            print(f"Average rating: {aggs_results['aggregations']['avg_rating']['value']:.2f}")
        
        if "price_ranges" in aggs_results["aggregations"]:
            print("\nPrice Range Distribution:")
            for bucket in aggs_results["aggregations"]["price_ranges"]["buckets"]:
                print(f"  {bucket['key']}: {bucket['doc_count']} restaurants")
        
        if "cuisines" in aggs_results["aggregations"]:
            print("\nTop Cuisines:")
            for bucket in aggs_results["aggregations"]["cuisines"]["buckets"]:
                print(f"  {bucket['key']}: {bucket['doc_count']} restaurants")
    except Exception as e:
        logger.warning(f"Could not get aggregate statistics: {e}")
    
    await es.close()

async def _main():
    parser = argparse.ArgumentParser(description="Inspect Elasticsearch index")
    parser.add_argument("--es_uri", type=str, help="The Elasticsearch URI", default="http://localhost:9200")
    parser.add_argument("--es_username", type=str, help="The Elasticsearch username", default="elastic")
    parser.add_argument("--es_password", type=str, help="The Elasticsearch password", default="yelp123")
    parser.add_argument("--es_index_name", type=str, help="The name of the Elasticsearch index", default="yelp_index")
    
    args = parser.parse_args()
    
    await inspect_index(args.es_uri, args.es_username, args.es_password, args.es_index_name)

if __name__ == "__main__":
    asyncio.run(_main()) 