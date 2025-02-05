from elasticsearch import Elasticsearch
from ..data_types import Restaurant, Result
from typing import Optional, List

class ESClient: 
    def __init__(self, uri: str, username: str, password: str):
        self.es_client = Elasticsearch(
            uri,
            basic_auth=(username, password)
        )

    def create_index(self, index: str):
        self.es_client.indices.create(index=index)
    
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
        self.es_client.bulk(
            index=index,
            body=[restaurant.model_dump() for restaurant in restaurants],
        )

    def delete_restaurant(self, index: str, restaurant_id: str):
        self.es_client.delete(index=index, id=restaurant_id)

    def delete_restaurant_bulk(self, index: str, restaurant_ids: List[str]):
        self.es_client.bulk(
            index=index,
            body=[{"_id": restaurant_id} for restaurant_id in restaurant_ids],
            delete=True,
        )

    def get_restaurant_by_id(self, index: str, restaurant_id: str) -> Optional[Restaurant]:
        response = self.es_client.get(index=index, id=restaurant_id)
        return Restaurant(**response["_source"])

    def pull_similar_restaurants(self, index: str, type: str, embedding: List[float], k: int = 10) -> List[Result]:
        embedding_field = 'text_embedding' if type == 'text' else 'photo_embedding'
        response = self.es_client.search(
            index=index,
            body={
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": f"cosineSimilarity(params.query_vector, '{embedding_field}') + 1.0",
                            "params": {"query_vector": embedding}
                        }
                    }
                },
                "size": k
            }
        )
        return [
            Result(
                restaurant=Restaurant(**hit["_source"]),
                score=hit["_score"],
            )
            for hit in response["hits"]["hits"]
        ]

def connect_es_client(uri: str, username: str, password: str):
    return ESClient(uri, username, password)



