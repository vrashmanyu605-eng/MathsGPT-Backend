import os
import json
from typing import List, Dict, Any, Tuple
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusException
)

# Constants
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "mathscare_embeddings"
DIMENSION = 768  # text-embedding-004 output dimension

def connect_milvus():
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise e

def create_collection_if_not_exists():
    connect_milvus()
    
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)
    
    print(f"Creating collection '{COLLECTION_NAME}'...")
    
    # Define fields
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535), # Adjust max_length as needed
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]
    
    schema = CollectionSchema(fields, "MathsCare document embeddings")
    
    collection = Collection(COLLECTION_NAME, schema)
    
    # Create index for faster search
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    
    return collection

def insert_embeddings_milvus(filename: str, content: str, embedding: List[float], metadata: Dict[str, Any]):
    try:
        collection = create_collection_if_not_exists()
        
        # Data to insert
        # pymilvus expects data as a list of lists/columns if using insert
        # [ [filename], [content], [embedding], [metadata] ]
        
        data = [
            [filename],
            [content],
            [embedding],
            [metadata]
        ]
        
        collection.insert(data)
        collection.flush() # Ensure data is visible
        print(f"Inserted embedding for {filename} into Milvus.")
        
    except Exception as e:
        print(f"Error inserting into Milvus: {e}")
        raise e

def search_similar_embeddings_milvus(query_embedding: List[float], top_k: int = 3) -> List[Tuple]:
    try:
        collection = create_collection_if_not_exists()
        collection.load() # Ensure loaded
        
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["filename", "content", "metadata"]
        )
        
        # Format results to match the expected format in app/training.py
        # Expected: List of tuples (filename, content, metadata)
        formatted_results = []
        for hits in results:
            for hit in hits:
                filename = hit.entity.get("filename")
                content = hit.entity.get("content")
                metadata = hit.entity.get("metadata")
                # distance = hit.distance # Not used in current return signature
                
                formatted_results.append((filename, content, metadata))
                
        return formatted_results
        
    except Exception as e:
        print(f"Error searching Milvus: {e}")
        raise e
