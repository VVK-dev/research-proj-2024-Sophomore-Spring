from OpenAI_utils import get_embedding, num_tokens_from_string
from Pinecone_utils import insert_vector_into_pinecone_index

#Split file into chunks and load into filetext

def get_data_from_file(filepath: str) -> list[str]:

    with open( filepath, "r") as file:
        
        filetext = file.read().strip().split("\n\n") 
        #assuming text file is written as paragraphs with one line between each para
        
        #NOTE: Dataset has not been chosen yet, so its format and thus this method is subject to change
        
        #the filetext variable now contains a list of chunks
        
        return filetext


#Get and insert vector for each chunk into pinecone index

def insert_vectors_from_data(filetext : list[str]):        
    
    #filetext is the list of all chunks
    
    for i in range(0, len(filetext)):
        
        #TODO: Add a short time gap between each request to reduce chances of hitting rate limit
        
        vector_val = get_embedding(filetext[i])
        
        #use index of chunk as id and its vector as vector_val to create entry into vector index in proper format
        
        vector = {"id" : str(i), "values" : vector_val}
        
        #TODO: Add a short time gap between each request to reduce chances of hitting rate limit
        
        insert_vector_into_pinecone_index(vector)
        
        
#Method to calculate Costs
def CalculateCosts(Filechunks : list[str] = None, Filechunk :str = None, isLlama2 : bool = False) -> float:
    
    num_tokens : int = 0
    
    if(Filechunks is None):
        
        num_tokens += num_tokens_from_string(Filechunk)
    
    else:
        
        for chunk in Filechunks:
            
            num_tokens += num_tokens_from_string(string = chunk)
    
    if(isLlama2):
        
        return (num_tokens * 0.0000002)
    
    else:
        
        #If not Llama 2 then text-embedding-3-small
        
        return (num_tokens * 0.00000002)