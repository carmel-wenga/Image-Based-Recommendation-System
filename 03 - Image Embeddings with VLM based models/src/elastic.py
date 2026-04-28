from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from .embeddings import ImageEmbeddingExtractor
from .settings import MODEL_FIELD_MAPPING, DATASET_DIR

import os

def update_index_mapping(es, index_name, body):
    """
    Updates the index mapping to add new fields if they don't exist.

    :param es: Elasticsearch client instance
    :param index_name: Name of the index
    :param body: Mapping definition (body for put_mapping)
    """
    es.indices.put_mapping(index=index_name, body=body)
    print(f"[INFO] Index '{index_name}' mapping updated.")



def bulk_update_embeddings(es, index_name, items, embeddings, model="resnet"):
    """
    Bulk updates the embeddings for the given model.
    """
    field_name = MODEL_FIELD_MAPPING.get(model, "resnet")
    actions = []

    for item, vector in zip(items, embeddings):
        actions.append({
            "_op_type": "update",
            "_index": index_name,
            "_id": item["_id"],
            "doc": {
                field_name: vector.flatten().tolist()
            }
        })

    bulk(es, actions)
    print(f"[INFO] Updated {len(actions)} documents with {model} embeddings.")


def extract_embeddings_and_update_es(
    es: Elasticsearch,
    embedding_extractor: ImageEmbeddingExtractor,
    model: str = "resnet",
    index_name: str = "items",
    batch_size=128,
    dataset_dir: str = DATASET_DIR,
):
    # query ES to get all the image path
    query = {
        "size": batch_size,
        "query": {
            "bool": {
                "must_not": {
                    "exists": {"field": MODEL_FIELD_MAPPING.get(model)}
                }
            }
        }
    }

    response = es.search(index=index_name, body=query, scroll="5m")
    scroll_id = response["_scroll_id"]

    while True:
        items = response["hits"]["hits"]

        if not items:
            break

        # batch processing
        paths = [os.path.join(dataset_dir, item["_source"]["imPath"]) for item in items]
        embeddings = embedding_extractor.extract_embeddings(paths)
        bulk_update_embeddings(es, index_name, items, embeddings, model)

        # continue scrolling
        response = es.scroll(scroll_id=scroll_id, scroll="5m")

    print("DONE: all embeddings updated.")


def knn_search(es, item_id, index_name, model="resnet", k=10, num_candidates=100, apply_filter=False):
    """
    Performs a KNN search using the specified model's embeddings.
    """
    # Query Elasticsearch to get all the fields of the referenced item
    res = es.get(index=index_name, id=item_id)
    ref_item = res['_source']

    # Determine which field to use based on the model
    field_name = MODEL_FIELD_MAPPING.get(model, "resnet")

    # Check if the reference item has the required embedding
    if field_name not in ref_item:
        raise ValueError(f"The reference item does not have embeddings for field '{field_name}'.")

    # build the knn query. Add a filter on the category if apply_filter is enabled
    knn_query = {
        "knn": {
            "field": field_name,
            "query_vector": ref_item[field_name],
            "k": k+1,  # +1 to account for the reference item itself being in the results
            "num_candidates": num_candidates
        }
    }

    if apply_filter:
        knn_query["knn"]["filter"] = {
            "term": {
                "category": ref_item['category'][0]
            }
        }

    # execute the knn query
    res = es.search(index=index_name, query=knn_query)
    knn_items = res['hits']['hits']
    return ref_item, knn_items[1:]  # Exclude the reference item itself from the results


def visual_search(
    es: Elasticsearch,
    src_vector: list,
    index_name: str = "items",
    model: str = "resnet",
    k:int =10,
    num_candidates:int = 100
):
    # build the knn query.
    knn_query = {
        "knn": {
            "field": MODEL_FIELD_MAPPING.get(model),
            "query_vector": src_vector,
            "k": k,
            "num_candidates": num_candidates
        }
    }

    # execute the knn query
    res = es.search(index=index_name, query=knn_query)
    knn_results = res['hits']['hits']
    return knn_results

def search_and_display(
    es: Elasticsearch,
    index_name: str = "items",
    model: str = "resnet",
    k: int = 10,
    num_candidates: int = 100
):
    pass
