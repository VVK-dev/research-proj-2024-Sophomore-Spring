from OpenAI_utils import get_embedding
from Pinecone_utils import insert_vector_into_pinecone_index

global filetext

#Split file into chunks and load into filetext

def get_data_from_file(filepath: str):

    with open( filepath, "r") as file:
        
        filetext = file.read().strip().split("\n\n") #assuming text file is written as paragraphs with one line 
        #between each para
        
        #the filetext variable now contains a list of chunks


#Get and insert vector for each chunk into pinecone index

def insert_vectors_of_data():        
    
    for i in range(0, len(filetext)):
        
        vector_val = get_embedding(filetext[i])
        
        #use id and vector_val to create entry into vector index in proper format
        
        vector = {"id" : str(i), "values" : vector_val}
        
        insert_vector_into_pinecone_index(vector)