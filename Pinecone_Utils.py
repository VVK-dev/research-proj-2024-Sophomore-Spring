import os
from pinecone import Pinecone, PodSpec

pinecone_client = Pinecone(api_key = os.getenv("PINECONE_API_KEY"))
    
index_name = "RAG-project-vectors"

#Check if starter index already exists

def index_exists() -> bool:
    
    if index_name in pinecone_client.list_indexes().names():
        
        return True
    else:
        
        return False

#Create starter index

def create_pinecone_index():

    if index_name not in pinecone_client.list_indexes().names(): #2nd check for redundancy

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

#Query the index

def query_pinecone_index(input_vector: list[float]) -> list[str]:
    
    matches : list[dict] = pinecone_client.Index("RAG-project-vectors").query(
        
        vector = input_vector,
        top_k = 3,        
    ).get("matches") #This will return a list of dictionaries containing the IDs and similarity score of the top 3 
    #matches
    
    #now get list of ids
    
    ids = list[str]
    
    for match in matches:
        
        ids.append(match.get("id"))
    
    return ids