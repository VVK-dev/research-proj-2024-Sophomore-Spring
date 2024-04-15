import os
from dotenv import load_dotenv, find_dotenv
import OpenAI_utils
import Neo4j_utils_LOCAL
from langchain_community.graphs import Neo4jGraph

#Initailize environment variables
_ = load_dotenv(find_dotenv(filename = "Keys.env"))

#Load Keys

Neo4j_URI = os.getenv("NEO4J_URI_LOCAL")
Username, Password = os.getenv("NEO4J_USERNAME_LOCAL"), os.getenv("NEO4J_PASSWORD_LOCAL")
OpenAIKey = os.getenv("OPENAI_API_KEY")

#Load Graph

knowledge_graph = Neo4jGraph(url = Neo4j_URI, username = Username, password = Password, database = "neo4j")

#Step 1: Create vector index if it doesn't already exist

Neo4j_utils_LOCAL.create_neo4j_vector_index(knowledge_graph)

#Step 2: Populate vector index if empty

Neo4j_utils_LOCAL.populate_neo4j_vector_index(knowledge_graph = knowledge_graph, OpenAIKey = OpenAIKey)

#Step 3: Get prompts

#1st set of prompts - Overall descriptive questions

descibe_prompt1 : str = "Tell me about the Barbie movie." 
descibe_prompt2 : str = "What is 'Barbenheimer'?"
descibe_prompt3 : str = "Describe the plot of the Barbie movie to me."
descibe_prompt4 : str = "Tell me about the development of the Barbie movie."
descibe_prompt5 : str = "Tell me about the writing of the Barbie movie."
descibe_prompt6 : str = "Tell me about the set design of the Barbie movie."
descibe_prompt7 : str = "Tell me about the music of the Barbie movie."
descibe_prompt8 : str = "How was the Barbie movie received at the box office?"
descibe_prompt9 : str = "Tell me how the Barbie movie was received by critics."
descibe_prompt10 : str = "Tell me about the themes the Barbie movie explores."

#2nd set of prompts - Relationship based questions

relation_prompt1 : str = "What did journalists from the New York Times say about the Barbie movie?"
relation_prompt2 : str = "How did Mattel influence the development of the Barbie movie?"
relation_prompt3 : str = "How does the main character of the Barbie movie change over the course of the film?"
relation_prompt4 : str = "List all the actresses that were offered to play as Stereotypical Barbie in the Barbie movie."
relation_prompt5 : str = "How did Sony Pictures influence the development of the Barbie movie?"
relation_prompt6 : str = "List all the things Greta Gerwig describes as influences for the story and development of the movie."
relation_prompt7 : str = "Which countries took action on the movie, regarding the nine dash line controversy?"
relation_prompt8 : str = "What did media outlets and newspapers say about the Barbie movie?"
relation_prompt9 : str = "What other Warner Bros. films did Barbie surpass in earnings?"
relation_prompt10 : str = "How does the character of 'Beach Ken' change over the course of the movie?"

prompts_with_context : dict[str,str] = {
    descibe_prompt1 : None, descibe_prompt2 : None, descibe_prompt3 : None, descibe_prompt4 : None,
    descibe_prompt5: None, descibe_prompt6 : None, descibe_prompt7 : None, descibe_prompt8 : None, 
    descibe_prompt9 : None, descibe_prompt10 : None, relation_prompt1 : None, relation_prompt2 : None,
    relation_prompt3 : None, relation_prompt4 : None, relation_prompt5 : None, relation_prompt6 : None,
    relation_prompt7 : None, relation_prompt8 : None, relation_prompt9 : None, relation_prompt10 : None    
    }

#Step 4: Get context for prompts

for prompt in prompts_with_context.keys():
    
    context = Neo4j_utils_LOCAL.search_neo4j_vector_index(knowledge_graph = knowledge_graph, OpenAIKey = OpenAIKey, prompt = prompt)
    
    #Sub-step 1 - update prompts_with_context dictionary
    
    prompts_with_context.update( {prompt: context} )

#Step 5: Send prompt with context to GPT 3.5 Turbo

responses_file = open(file = os.getenv("TEST_3_GRAPH_RESPONSES"), mode = 'a', encoding = 'UTF-8')

for prompt, context in prompts_with_context.items():
    
    prompt_with_context : str = f"""Respond to the following prompt using the context given below it.\nPrompt: {prompt} \nContext: {context}"""

    result : str = OpenAI_utils.get_response(prompt = prompt_with_context, model = "gpt-3.5-turbo", is_question = True)
        
    responses_file.write(f"PROMPT: \n{prompt_with_context}\n RESPONSE: \n{result}\n")
    responses_file.write("---------------------------------------------------------\n")
    

responses_file.close()