from elasticsearch import Elasticsearch
from rec_engine.core.agent import LLMClient
from rec_engine.data_types import Restaurant, Result
from typing import Optional, List, Dict, Any
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
                refresh=True
            )
            
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
    
    async def search_restaurants(self, index: str, query: str, k: int = 10, llm_client=None, keywords=None, cross_modal= False) -> List[Result]:
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
        
        if not self.index_exists(index):
            logging.error(f"Index {index} does not exist!")
            return []
        
        is_image_path = query.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        is_file_path = os.path.exists(query) and os.path.isfile(query)
        
        
        if is_image_path and not is_file_path:
            error_msg = f"Image file not found: {query}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        is_image_query = is_image_path and is_file_path
        
        logging.info(f"Processing {'image' if is_image_query else 'text'} query: {query}")
        
        if is_image_query:
            try:
                with open(query, "rb") as img_file:
                    image_data = img_file.read()
                embedding = await llm_client.get_embedding(image_data, "image")
                query_type = "photo"
            except Exception as e:
                error_msg = f"Error processing image file: {e}"
                logging.error(error_msg)
                raise RuntimeError(error_msg)
        else:
            # It's a text query
            embedding = await llm_client.get_embedding(query, "text")
            query_type = "text"
        
        logging.info(f"Generated embedding with {len(embedding)} dimensions")
        logging.info(f"Using query type: {query_type}")
        
        # Search using the generated embedding
        if cross_modal:
            results = self.pull_similar_restaurants_cross(index, query_type, embedding, k, keywords)
        else:
            results = self.pull_similar_restaurants(index, query_type, embedding, k, keywords)
        logging.info(f"Found {len(results)} results")
        
        return results
    
    def pull_similar_restaurants_cross(self, index: str, type: str, embedding: List[float], k: int = 10, keywords: Dict[str, Any] = None) -> List[Result]:

        print(f"Performing cross-modal search with {type} input")
        if keywords:
            print(f"Filtering with keywords: {keywords}")
        
        script_source = """
        double max_score = 0;
        boolean found_embedding = false;
        
        // Search through text embeddings
        for (int i = 0; i < 10; i++) {
            String text_field = 'text_embedding_' + i;
            if (doc.containsKey(text_field)) {
                found_embedding = true;
                double score = cosineSimilarity(params.query_vector, text_field) + 1.0;
                if (score > max_score) {
                    max_score = score;
                }
            }
        }
        
        // Search through photo embeddings
        for (int i = 0; i < 50; i++) {
            String photo_field = 'photo_embedding_' + i;
            if (doc.containsKey(photo_field)) {
                found_embedding = true;
                double score = cosineSimilarity(params.query_vector, photo_field) + 1.0;
                if (score > max_score) {
                    max_score = score;
                }
            }
        }
        
        // Return 0 if no embeddings were found
        return found_embedding ? max_score : 0;
        """
        
        if keywords and isinstance(keywords, dict):
            query = {
                "bool": {
                    "must": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": script_source,
                                    "params": {"query_vector": embedding}
                                }
                            }
                        }
                    ],
                    "filter": []
                }
            }
            
            if "cuisine" in keywords and keywords["cuisine"] != "None":
                cuisine_filter = {
                    "match": {
                        "cuisine": keywords["cuisine"]
                    }
                }
                query["bool"]["filter"].append(cuisine_filter)
                
            if "price_range" in keywords and keywords["price_range"] != "None":
                price_filter = {
                    "term": {
                        "price_range": keywords["price_range"]
                    }
                }
                query["bool"]["filter"].append(price_filter)
                
            if "rating" in keywords and keywords["rating"] != -1:
                rating_filter = {
                    "range": {
                        "rating": {
                            "gte": keywords["rating"]
                        }
                    }
                }
                query["bool"]["filter"].append(rating_filter)
        else:
            query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": script_source,
                        "params": {"query_vector": embedding}
                    }
                }
            }
        
        try:
            print(f"Executing cross-modal Elasticsearch query with {len(embedding)} dimensional vector")
            response = self.es_client.search(
                index=index,
                body={
                    "query": query,
                    "size": k
                }
            )
            
            print(f"Elasticsearch response received with {len(response['hits']['hits'])} hits")
            
            results = [
                Result(
                    restaurant=Restaurant(**hit["_source"]),
                    score=hit["_score"],
                )
                for hit in response["hits"]["hits"]
                if hit["_score"] > 0
            ]
            
            print(f"After filtering, {len(results)} results remain")
            return results
            
        except Exception as e:
            print(f"Error in Elasticsearch query: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def pull_similar_restaurants(self, index: str, type: str, embedding: List[float], k: int = 10, keywords: Dict[str, Any] = None) -> List[Result]:
        embedding_field_prefix = 'text_embedding_' if type == 'text' else 'photo_embedding_'
        
        print(f"Searching for {type} embeddings with prefix: {embedding_field_prefix}")
        if keywords:
            print(f"Filtering with keywords: {keywords}")
        
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
        
        if keywords and isinstance(keywords, dict):
            query = {
                "bool": {
                    "must": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": script_source,
                                    "params": {"query_vector": embedding}
                                }
                            }
                        }
                    ],
                    "filter": []
                }
            }
            
            if "cuisine" in keywords and keywords["cuisine"] != "None":
                cuisine_filter = {
                    "match": {
                        "cuisine": keywords["cuisine"]
                    }
                }
                query["bool"]["filter"].append(cuisine_filter)
                
            if "price_range" in keywords and keywords["price_range"] != "None":
                price_filter = {
                    "term": {
                        "price_range": keywords["price_range"]
                    }
                }
                query["bool"]["filter"].append(price_filter)
                
            if "rating" in keywords and keywords["rating"] != -1:
                rating_filter = {
                    "range": {
                        "rating": {
                            "gte": keywords["rating"]
                        }
                    }
                }
                query["bool"]["filter"].append(rating_filter)
        else:
            query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": script_source,
                        "params": {"query_vector": embedding}
                    }
                }
            }
        
        try:
            print(f"Executing Elasticsearch query with {len(embedding)} dimensional vector")
            response = self.es_client.search(
                index=index,
                body={
                    "query": query,
                    "size": k
                }
            )
            
            print(f"Elasticsearch response received with {len(response['hits']['hits'])} hits")
            
            results = [
                Result(
                    restaurant=Restaurant(**hit["_source"]),
                    score=hit["_score"],
                )
                for hit in response["hits"]["hits"]
                if hit["_score"] > 0
            ]
            
            print(f"After filtering, {len(results)} results remain")
            return results
            
        except Exception as e:
            print(f"Error in Elasticsearch query: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

def connect_es_client(uri: str, username: str, password: str):
    return ESClient(uri, username, password)



