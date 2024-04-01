import os
import sys
from dotenv import load_dotenv, find_dotenv
import Llama2_utils
import OpenAI_utils
import Pinecone_utils
import Dataset_utils
import urllib.parse
import time

#Initailize environment variables
_ = load_dotenv(find_dotenv(filename = "Keys.env"))

'''RAG System'''

#Step 1: Get article names from file, each name is one chunk

filenames : list[str] = Dataset_utils.get_data_from_file(os.getenv("ARTICLES_PATH"))

#Sub-step 1 - confirm procedure after showing costs for chunk embeddings

if(input(f"Getting embeddings for the file names will cost: {Dataset_utils.CalculateCosts(filenames)} if the index does not already exist. Proceed?") != "Y"):
    
    sys.exit(0)
    
#Step 2: Check if index exists

if (not Pinecone_utils.index_exists()):
    
    #If the index doesn't exist, create it
    
    Pinecone_utils.create_pinecone_index()
    
    #Populate the index once created
    
    for name in filenames:
        
        #decode name of each article and remove underscores to make text embeddings better
        name = urllib.parse.unquote(name).replace("_"," ") 
    
    Pinecone_utils.insert_vectors_from_data(filetext = filenames)

#Step 3: Get prompts

prompt1 : str = "Tell me about <SOMETHING>." #Overall descriptive question
prompt2 : str = "<SECOND PROMPT>"
prompt3 : str = "<THIRD PROMPT>"
prompt4 : str = "Ask me a question about <SOMETHING>." #Checking the model's understanding of the text
prompt5 : str = "Ask me a question about <SOMETHING> and <SOMETHING>." #Checking the model's understanding of multiple texts together

prompts_with_context : dict[str,str] = {prompt1 : None, prompt2 : None, prompt3 : None, prompt4 : None, prompt5: None}

#Step 4: Get context for prompt 

for prompt in prompts_with_context.keys():
    
    #Sub-step 1 - get vector embedding of prompt 
    
    vector : list[float] = OpenAI_utils.get_embedding(prompt)
    
    #Sub-step 2 - query over index
    
    matching_ids : list[str] = Pinecone_utils.query_pinecone_index(vector)
    
    #Sub-step 3 - get context from wikipedia article
    
    context : str = ""
    
    for index in matching_ids:
        
        article_path = os.getenv("WIKI_ARTICLES_FOLDER_PATH") + "//" + filenames[int(index)] + ".txt"
        
        with open(file = article_path, mode = "r") as wiki_article:
            context += wiki_article.read()
    
    #Sub-step 4 - update prompt with context
    
    prompts_with_context.update({prompt : context})

#Step 5: Send prompt to Llama with context

responses_file = open(file = os.getenv("TEXT_RESPONSES"), mode = 'a', encoding = 'UTF-8')

for prompt, context in prompts_with_context.items():
    
    prompt_with_context : str = f"Respond to the following prompt using the context given below it.\n 
    Prompt: {prompt} \n Context: {context}"

    time.sleep(1.0) #wait 1s to avoid being rate limited by together.ai
    
    result : str = Llama2_utils.llama(prompt = prompt_with_context)
    
    #Possible consideration: use llama_chat() instead and keep track of conversation?
    
    responses_file.write(f"PROMPT: \n{prompt_with_context}\n RESPONSE: \n{result}\n")
    responses_file.write("---------------------------------------------------------")
    
responses_file.close()