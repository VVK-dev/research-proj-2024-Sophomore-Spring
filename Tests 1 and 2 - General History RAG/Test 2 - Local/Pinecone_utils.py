import os
from pinecone import Pinecone, PodSpec
from OpenAI_utils import get_embedding
from dotenv import load_dotenv, find_dotenv
import time

#Initailize environment variables
_ = load_dotenv(find_dotenv(filename = "Keys.env"))


pinecone_client = Pinecone(api_key = os.getenv("PINECONE_API_KEY"))
    
index_name = "rag-project-vectors"

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
            dimension = 1536,
            metric = "cosine",
            spec = PodSpec(environment = "gcp-starter")
        )
        
    # wait for index to be initialized
    while not pinecone_client.describe_index(index_name).status['ready']:
        
        time.sleep(1)

#Get and insert vector for each chunk into pinecone index

def insert_vectors_from_data(filetext : list[str]):        
    
    openai_embedding_prompt_counter : int = 0
    
    index = pinecone_client.Index(index_name)
    
    #filetext is the list of all chunks
    
    for i in range(0, len(filetext)):
        
        if(openai_embedding_prompt_counter >= 3000):
            
            #if rate limit hit, wait 1 minute to refresh it 
            
            time.sleep(60)
            openai_embedding_prompt_counter = 0
        
        vector_val = get_embedding(filetext[i])
        
        openai_embedding_prompt_counter += 1
        
        #use index of chunk as id and its vector as vector_val to create entry into vector index in proper format
        
        vector = {"id" : str(i), "values" : vector_val}
        
        #TODO: Add a short time gap between each request to reduce chances of hitting rate limit
        
        index.upsert(
        
            vectors = [vector]
        )


#Query the index

def query_pinecone_index(input_vector: list[float]) -> list[str]:
    
    matches : list[dict] = pinecone_client.Index(index_name).query(
        
        vector = input_vector,
        top_k = 1,        
    ).get("matches") #This will return a list of dictionaries containing the IDs and similarity score of the top 3 
    #matches
    
    #now get list of ids
    
    ids : list[str] = []
    
    for match in matches:
        
        ids.append(match.get("id"))
    
    return ids