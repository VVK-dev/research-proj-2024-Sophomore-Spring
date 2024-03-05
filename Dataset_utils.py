from OpenAI_utils import get_embedding
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
    
    for i in range(0, len(filetext)):
        
        vector_val = get_embedding(filetext[i])
        
        #use index of chunk as id and its vector as vector_val to create entry into vector index in proper format
        
        vector = {"id" : str(i), "values" : vector_val}
        
        insert_vector_into_pinecone_index(vector)