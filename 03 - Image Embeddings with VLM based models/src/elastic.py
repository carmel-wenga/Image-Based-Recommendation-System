from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Map the model parameter to the correct Elasticsearch field
MODEL_FIELD_MAPPING = {
    "resnet": "image_features",
    "siglip": "image_features_siglip",
    "vertex_api": "image_features_vertex_api"
}

def create_index_with_mapping(es, index_name):
    """
    Creates the Elasticsearch index with mappings for all three types of embeddings.
    """
    mapping = {
        "mappings": {
            "properties": {
                "image_features": { # ResNet50 (2048 dims)
                    "type": "dense_vector",
                    "dims": 2048,
                    "index": True,
                    "similarity": "cosine"
                },
                "image_features_siglip": { # SigLIP base patch16 224 (768 dims)
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine"
                },
                "image_features_vertex_api": { # Vertex AI Multimodal Embedding (1408 dims)
                    "type": "dense_vector",
                    "dims": 1408,
                    "index": True,
                    "similarity": "cosine"
                },
                "category": {
                    "type": "keyword"
                },
                "imPath": {
                    "type": "keyword"
                }
            }
        }
    }
    
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)
        print(f"[INFO] Index '{index_name}' created with mapping.")
    else:
        print(f"[INFO] Index '{index_name}' already exists.")

def update_index_mapping(es, index_name, query):
    """
    Updates the index mapping to add new fields if they don't exist.

    :param es: Elasticsearch client instance
    :param index_name: Name of the index
    :param query: Mapping definition (body for put_mapping)
    """
    es.indices.put_mapping(index=index_name, body=query)
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
