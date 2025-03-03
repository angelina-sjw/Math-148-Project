from elasticsearch import Elasticsearch
from rec_engine.core.agent import LLMClient
from rec_engine.data_types import Restaurant, Result
from typing import Optional, List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class ESClient: 
    def __init__(self, uri: str, username: str, password: str):
        self.es_client = Elasticsearch(
            uri,
            basic_auth=(username, password)
        )

    def create_index(self, index: str):
        mapping = {
            "mappings": {
                "properties": {
                    "business_id": {"type": "keyword"},
                    "name": {"type": "text"},
                    "address": {"type": "text"},
                    "cuisine": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "rating": {"type": "float"},
                    "price_range": {"type": "keyword"},
                    "profile_text_list": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "photo_ids": {"type": "keyword"},
                    "captions": {"type": "text"},
                    "labels": {"type": "keyword"},
                    # Dynamic fields for embeddings with numeric suffixes
                    "text_embedding_*": {
                        "type": "dense_vector",
                        "dims": 1536,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "photo_embedding_*": {
                        "type": "dense_vector",
                        "dims": 512,
                        "index": True,
                        "similarity": "cosine"
                    }
                },
                "dynamic_templates": [
                    {
                        "embedding_fields": {
                            "match": "text_embedding_*",
                            "mapping": {
                                "type": "dense_vector",
                                "dims": 512,
                                "index": True,
                                "similarity": "cosine"
                            }
                        }
                    },
                    {
                        "photo_embedding_fields": {
                            "match": "photo_embedding_*",
                            "mapping": {
                                "type": "dense_vector",
                                "dims": 512,
                                "index": True,
                                "similarity": "cosine"
                            }
                        }
                    }
                ]
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "30s",
                "index.mapping.nested_fields.limit": 100,
                "index.mapping.total_fields.limit": 2000
            }
        }
        
        # Create the index with the mapping
        self.es_client.indices.create(index=index, body=mapping)
    
    def delete_index(self, index: str):
        self.es_client.indices.delete(index=index)

    def index_exists(self, index: str):
        return self.es_client.indices.exists(index=index)
    
    def add_restaurant(self, index: str, restaurant: Restaurant):
        self.es_client.index(
            index=index,
            body=restaurant.model_dump(),
        )

    def add_restaurant_bulk(self, index: str, restaurants: List[Restaurant]):
        bulk_data = []
        for restaurant in restaurants:
            bulk_data.append({"index": {"_index": index}})
            bulk_data.append(restaurant.model_dump())
        
        if bulk_data:
            response = self.es_client.bulk(
                body=bulk_data,
                refresh=True  # Force refresh
            )
            
            # Check for errors
            if response.get('errors', False):
                error_items = [item for item in response['items'] if 'error' in item['index']]
                logging.error(f"Bulk indexing had {len(error_items)} errors: {error_items[:3]}")
            else:
                logging.info(f"Successfully indexed {len(restaurants)} documents")

    def delete_restaurant(self, index: str, restaurant_id: str):
        self.es_client.delete(index=index, id=restaurant_id)

    def delete_restaurant_bulk(self, index: str, restaurant_ids: List[str]):
        bulk_data = []
        for restaurant_id in restaurant_ids:
            bulk_data.append({"delete": {"_index": index, "_id": restaurant_id}})
        if bulk_data:
            self.es_client.bulk(body=bulk_data)

    def get_restaurant_by_id(self, index: str, restaurant_id: str) -> Optional[Restaurant]:
        response = self.es_client.get(index=index, id=restaurant_id)
        return Restaurant(**response["_source"])
    
    async def search_restaurants(self, index: str, query: str, k: int = 10, llm_client=None, keywords=None) -> List[Result]:
        """
        Search for restaurants using either text query or path to a photo.
        
        Args:
            index: The Elasticsearch index name
            query: Either a text query or a file path to an image
            k: Number of results to return
            llm_client: LLMClient instance for generating embeddings
            keywords: Optional extracted keywords for advanced search
            
        Returns:
            List of restaurant results
        """
        if llm_client is None:
            raise ValueError("LLMClient is required for generating embeddings")
        
        # Check if the index exists
        if not self.index_exists(index):
            logging.error(f"Index {index} does not exist!")
            return []
        
        # Determine if query is a file path or text
        is_file_path = os.path.exists(query) and os.path.isfile(query)
        
        logging.info(f"Processing {'image' if is_file_path else 'text'} query: {query}")
        
        if is_file_path:
            # It's a file path, process as an image
            with open(query, "rb") as img_file:
                image_data = img_file.read()
            embedding = await llm_client.get_embedding(image_data, "image")
            query_type = "photo"
        else:
            # It's a text query
            embedding = await llm_client.get_embedding(query, "text")
            query_type = "text"
        
        logging.info(f"Generated embedding with {len(embedding)} dimensions")
        
        # Search using the generated embedding
        results = self.pull_similar_restaurants(index, query_type, embedding, k)
        logging.info(f"Found {len(results)} results")
        
        return results

    def pull_similar_restaurants(self, index: str, type: str, embedding: List[float], k: int = 10) -> List[Result]:
        embedding_field_prefix = 'text_embedding_' if type == 'text' else 'photo_embedding_'
        
        logger.info(f"Searching for {type} embeddings with prefix: {embedding_field_prefix}")
        
        # This script will find the maximum similarity across all embeddings of the specified type
        script_source = f"""
        double max_score = 0;
        boolean found_embedding = false;
        
        for (int i = 0; i < 10; i++) {{  // Assuming max 10 embeddings per restaurant
            String field = '{embedding_field_prefix}' + i;
            if (doc.containsKey(field)) {{
                found_embedding = true;
                double score = cosineSimilarity(params.query_vector, field) + 1.0;
                if (score > max_score) {{
                    max_score = score;
                }}
            }}
        }}
        
        // Return 0 if no embeddings were found
        return found_embedding ? max_score : 0;
        """
        
        try:
            logger.info(f"Executing Elasticsearch query with {len(embedding)} dimensional vector")
            response = self.es_client.search(
                index=index,
                body={
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": script_source,
                                "params": {"query_vector": embedding}
                            }
                        }
                    },
                    "size": k
                }
            )
            
            logger.info(f"Elasticsearch response received with {len(response['hits']['hits'])} hits")
            
            # Filter out results with zero score (no matching embeddings)
            results = [
                Result(
                    restaurant=Restaurant(**hit["_source"]),
                    score=hit["_score"],
                )
                for hit in response["hits"]["hits"]
                if hit["_score"] > 0
            ]
            
            logger.info(f"After filtering, {len(results)} results remain")
            return results
            
        except Exception as e:
            logger.error(f"Error in Elasticsearch query: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

def connect_es_client(uri: str, username: str, password: str):
    return ESClient(uri, username, password)



