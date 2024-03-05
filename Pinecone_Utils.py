import os
from pinecone import Pinecone, PodSpec

pinecone_client = Pinecone(api_key = os.getenv("PINECONE_API_KEY"))
    
index_name = "RAG-project-vectors"

#Create starter index

def create_pinecone_index():

    if index_name not in pinecone_client.list_indexes().names():

        pinecone_client.create_index(
        
            name = index_name,
            dimension = 768,
            metric = "cosine",
            spec = PodSpec(environment = "gcp-starter")
        )

#Insert a vector into the index

def insert_vector_into_pinecone_index(vector : dict):

    pinecone_client.Index(index_name).upsert(
        
        vectors = [vector]
    )

