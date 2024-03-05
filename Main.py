import os
from dotenv import load_dotenv, find_dotenv
import Llama2_utils
import OpenAI_utils
import Pinecone_utils
import Dataset_utils

#Initailize global variables
_ = load_dotenv(find_dotenv())

#Step 1: Get chunks from file

filechunks : list[str] = Dataset_utils.get_data_from_file(os.getenv("DATA_FILE_PATH"))

#Step 2: Check if index exists

if (not Pinecone_utils.index_exists()):
    
    #If the index doesn't exist, create it
    
    Pinecone_utils.create_pinecone_index()
    
    #Populate the index once created
    
    Dataset_utils.insert_vectors_from_data(filetext = filechunks)

#Step 3: Get prompts

prompt1 : str = "<FIRST PROMPT>"
prompt2 : str = "<SECOND PROMPT>"
prompt3 : str = "<THIRD PROMPT>"
prompt4 : str = "<FOURTH PROMPT>"
prompt5 : str = "<FIFTH PROMPT>"

prompts_with_context : dict[str,str] = {prompt1 : None, prompt2 : None, prompt3 : None, prompt4 : None, prompt5: None}

