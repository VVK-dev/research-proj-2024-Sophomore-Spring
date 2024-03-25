import os
import sys
from dotenv import load_dotenv, find_dotenv
import Llama2_utils
import OpenAI_utils
import Pinecone_utils
import Dataset_utils

#Initailize global variables
_ = load_dotenv(find_dotenv())

#Method to calculate Costs
def CalculateCosts(Filechunks : list[str], isLlama2 : bool = False) -> float:
    
    num_tokens : int = 0
    
    for chunk in Filechunks:
        
        num_tokens += OpenAI_utils.num_tokens_from_string(string = chunk)
    
    if(isLlama2):
        
        return (num_tokens * 0.0000002)
    
    else:
        
        return (num_tokens * 0.00000002)


'''RAG System'''

#Step 1: Get chunks from file

filechunks : list[str] = Dataset_utils.get_data_from_file(os.getenv("DATA_FILE_PATH"))

#Sub-step 1 - confirm procedure after showing costs for chunk embeddings

if(input(f"Getting embeddings for the file will cost: {CalculateCosts(filechunks)}. Proceed?") is not "Y"):
    
    sys.exit(0)
    
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

#Sub-step 3 - confirm procedure after showing costs for prompt embeddings

if(input(f"Getting embeddings for the prompts will cost: {CalculateCosts(Filechunks = list(prompts_with_context.keys()))}. Proceed?") 
   is not "Y"):
    
    sys.exit(0)

#Step 4: Get context for prompt 

for prompt in prompts_with_context.keys():
    
    #Sub-step 1 - get vector embedding of prompt 
    
    vector : list[float] = OpenAI_utils.get_embedding(prompt)
    
    #Sub-step 2 - query over index
    
    matching_ids : list[str] = Pinecone_utils.query_pinecone_index(vector)
    
    #Sub-step 3 - join matching chunks together to get context as a single string
    
    context : str = ""
    
    for id in matching_ids:
        
        context += (filechunks[int(id)])
    
    #Sub-step 4 - update prompt with context
    
    prompts_with_context.update({prompt : context})

#Step 5: Send prompt to Llama with context

#TODO: Add a short time gap between each request to reduce chances of hitting rate limit

for prompt, context in prompts_with_context.items():
    
    prompt_with_context : str = f"Respond to the following prompt using the context given below it.\n 
    Prompt: {prompt} \n Context: {context}"
    
    #Sub-step 1 - confirm procedure after showing costs
    
    #TODO: Clean this up, maybe move it out
    if(input(f"Prompting Llama 2 will cost: {CalculateCosts((prompts_with_context), True)}. Proceed?") is not "Y"):
    
        sys.exit(0)

    print(Llama2_utils.llama(prompt = prompt_with_context))
    
#Possible consideration: use llama_chat() instead and keep track of conversation?