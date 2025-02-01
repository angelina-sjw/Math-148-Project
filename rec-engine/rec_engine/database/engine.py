from elasticsearch import Elasticsearch
from ..data_types import Restaurant, Result
from typing import Optional, List, Float, String

class ESClient: 
    def __init__(self,uri: str, username: str, password: str):
        self.es_client = Elasticsearch(
            uri,
            basic_auth=(username, password)
        )

    def add_restaurant(self, restaurant: Restaurant):
        self.es_client.index(
            index="restaurants",
            body=restaurant.model_dump(),
        )

    async def add_restaurant_bulk(self, restaurants: List[Restaurant]):
        await self.es_client.bulk(
            index="restaurants",
            body=[restaurant.model_dump() for restaurant in restaurants],
        )

    async def delete_restaurant(self, restaurant_id: String):
        await self.es_client.delete(index="restaurants", id=restaurant_id)

    async def delete_restaurant_bulk(self, restaurant_ids: List[String]):
        await self.es_client.bulk(
            index="restaurants",
            body=[{"_id": restaurant_id} for restaurant_id in restaurant_ids],
            delete=True,
        )

    async def get_restaurant_by_id(self, restaurant_id: String) -> Optional[Restaurant]:
        response = await self.es_client.get(index="restaurants", id=restaurant_id)
        return Restaurant(**response["_source"])

    async def pull_similar_restaurants(self, embedding: list[Float], k: int = 10) -> List[Result]:
        response = await self.es_client.search(
            index="restaurants",
            body={
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
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



