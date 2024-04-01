import os
from dotenv import load_dotenv, find_dotenv
import Llama2_utils
import Neo4j_utils
from langchain_community.graphs import Neo4jGraph
from Dataset_utils import CalculateCosts
import sys
import time

#Initailize global variables
_ = load_dotenv(find_dotenv(filename = "Keys.env"))

#Load Keys

Neo4j_URI = os.getenv("NEO4J_URI")
Username, Password = os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")
OpenAIKey = os.getenv("OPENAI_API_KEY")

#Load Graph

knowledge_graph = Neo4jGraph(url = Neo4j_URI, username = Username, password = Password, database = "neo4j")

neo4j_query_counter : int = 0

#Step 1: Create vector index if it doesn't already exist

Neo4j_utils.create_neo4j_vector_index(knowledge_graph)

#Step 2: Populate vector index if empty

Neo4j_utils.populate_neo4j_vector_index(knowledge_graph = knowledge_graph, OpenAIKey = OpenAIKey)

#Step 3: Get prompts

prompt1 : str = "Tell me about <SOMETHING>." #Overall descriptive question
prompt2 : str = "<SECOND PROMPT>"
prompt3 : str = "<THIRD PROMPT>"
prompt4 : str = "Ask me a question about <SOMETHING>." #Checking the model's understanding of the text
prompt5 : str = "Ask me a question about <SOMETHING> and <SOMETHING>." #Checking the model's understanding of multiple texts together

prompts_with_context : dict[str,str] = {prompt1 : None, prompt2 : None, prompt3 : None, prompt4 : None, prompt5: None}

#Step 4: Get context for prompts

for prompt in prompts_with_context.keys():
    
    context = Neo4j_utils.search_neo4j_vector_index(knowledge_graph = knowledge_graph, OpenAIKey = OpenAIKey, prompt = prompt)
    
    prompts_with_context.update( {prompt: context} )

#Step 5: Send prompt with context to Llama

#NOTE: Change below env variable when using gpt 4
responses_file = open(file = os.getenv("GPT_3.5_RESPONSES"), mode = 'a', encoding = 'UTF-8')

for prompt, context in prompts_with_context.items():
    
    prompt_with_context : str = f"Respond to the following prompt using the context given below it.\n 
    Prompt: {prompt} \n Context: {context}"

    time.sleep(1.0) #wait 1s to avoid being rate limited by together.ai
    
    result : str = Llama2_utils.llama(prompt = prompt_with_context)
    
    #Possible consideration: use llama_chat() instead and keep track of conversation?
    
    responses_file.write(f"PROMPT: \n{prompt_with_context}\n RESPONSE: \n{result}\n")
    responses_file.write("---------------------------------------------------------")
    

responses_file.close()