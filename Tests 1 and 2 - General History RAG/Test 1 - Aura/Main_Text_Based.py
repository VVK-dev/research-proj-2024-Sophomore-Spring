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

#1st set of prompts - Overall descriptive questions

descibe_prompt1 : str = "Tell me about Édouard Manet." 
descibe_prompt2 : str = "Tell me about the 2004 Indian Ocean earthquake."
descibe_prompt3 : str = "Tell me about the 19th century."
descibe_prompt4 : str = "Tell me about the 16th century."
descibe_prompt5 : str = "Tell me about the 1973 Oil Crisis."
descibe_prompt6 : str = "Tell me about the 2004 Atlantic hurricane season."
descibe_prompt7 : str = "Tell me about the 2005 Atlantic hurricane season."
descibe_prompt8 : str = "Tell me about the 1st century."
descibe_prompt9 : str = "Tell me about the 1980 eruption of Mount St. Helens."
descibe_prompt10 : str = "Tell me about 1st century BC."

#2nd set of prompts - Checking the model's understanding of how multiple texts relate
relation_prompt1 : str = "How did Édouard Manet influence Europe?"
relation_prompt2 : str = "What type of paintings would Édouard Manet paint?"
relation_prompt3 : str = "Were there any relationships between the 2005 Indian Ocean earthquake and the South China Sea?"
relation_prompt4 : str = "How did the 1980 eruption of Mount St. Helens affect tourism?"
relation_prompt5 : str = "What happened to the Ottoman Empire in the 19th century?"
relation_prompt6 : str = "What was the 2005 Hertfordshire Oil Storage Terminal Fire adjacent to?"
relation_prompt7 : str = "What led to the Yom Kippur War?"
relation_prompt8 : str = "What major war preceded the 1973 oil crisis?"
relation_prompt9 : str = "What hurricane season did hurricane Epsilon belong to?"
relation_prompt10 : str = "Which countries did the 2005 Atlanic hurricane season affect?"

prompts_with_context : dict[str,str] = {
    descibe_prompt1 : None, descibe_prompt2 : None, descibe_prompt3 : None, descibe_prompt4 : None,
    descibe_prompt5: None, descibe_prompt6 : None, descibe_prompt7 : None, descibe_prompt8 : None, 
    descibe_prompt9 : None, descibe_prompt10 : None, relation_prompt1 : None, relation_prompt2 : None,
    relation_prompt3 : None, relation_prompt4 : None, relation_prompt5 : None, relation_prompt6 : None,
    relation_prompt7 : None, relation_prompt8 : None, relation_prompt9 : None, relation_prompt10 : None    
    }

#Step 4: Get context for prompt 

for prompt in prompts_with_context.keys():
    
    #Sub-step 1 - get vector embedding of prompt 
    
    vector : list[float] = OpenAI_utils.get_embedding(prompt)
    
    #Sub-step 2 - query over index
    
    matching_ids : list[str] = Pinecone_utils.query_pinecone_index(vector)
    
    #Sub-step 3 - get context from wikipedia article
    
    context : str = ""
    
    for index in matching_ids:
        
        article_path = os.getenv("WIKI_ARTICLES_FOLDER_PATH") + "\\" + filenames[int(index)] + ".txt"
        
        with open(file = article_path, mode = "r", encoding='UTF-8') as wiki_article:
            context += wiki_article.read()
    
    
    #Sub-step 4 - reduce size of context if it's too big
    
    context = Dataset_utils.token_chopper(context = context)
    
    #Sub-step 5 - update prompt with context
    
    prompts_with_context.update({prompt : context})

#Step 5: Send prompt with context to Llama

responses_file = open(file = os.getenv("TEST_1_TEXT_RESPONSES"), mode = 'a', encoding = 'UTF-8')

for prompt, context in prompts_with_context.items():
    
    prompt_with_context : str = f"""Respond to the following prompt using the context given below it.\nPrompt: {prompt} \nContext: {context}"""

    time.sleep(1.0) #wait 1s to avoid being rate limited by together.ai
    
    result : str = Llama2_utils.llama(prompt = prompt_with_context)
    
    #Possible consideration: use llama_chat() instead and keep track of conversation?
    
    responses_file.write(f"PROMPT: \n{prompt_with_context}\n RESPONSE: \n{result}\n")
    responses_file.write("---------------------------------------------------------\n")
    

responses_file.close()