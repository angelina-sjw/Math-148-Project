from elasticsearch import Elasticsearch


es_client = Elasticsearch(
    hosts=[ 
        "http://localhost:9200"
    ],
    basic_auth=("elastic", "password")
)

