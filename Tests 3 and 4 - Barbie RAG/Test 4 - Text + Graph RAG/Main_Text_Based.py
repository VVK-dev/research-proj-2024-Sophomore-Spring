import os
from dotenv import load_dotenv, find_dotenv
import OpenAI_utils
import Pinecone_utils
import Dataset_utils

#Initailize environment variables
_ = load_dotenv(find_dotenv(filename = "Keys.env"))

'''RAG System'''

#Step 1: Get paragraphs from file, each is one chunk

paragraph_chunks : list[str] = Dataset_utils.get_paragraphs_from_file(os.getenv("BARBIE_ARTICLE_TEXT_PATH"))

#Step 2: Check if index exists

if (not Pinecone_utils.index_exists()):
    
    #If the index doesn't exist, create it
    
    Pinecone_utils.create_pinecone_index()
    
    #Populate the index once created
    
    Pinecone_utils.insert_vectors_from_data(filetext = paragraph_chunks)

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
relation_prompt6 : str = "List all the things Greta Gerwig describes as influences for the story and development of the Barbie movie."
relation_prompt7 : str = "Which countries took action on the Barbie movie regarding the nine dash line controversy and why?"
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

#Step 4: Get context for prompt 

for prompt in prompts_with_context.keys():
    
    #Sub-step 1 - get vector embedding of prompt 
    
    vector : list[float] = OpenAI_utils.get_embedding(prompt)
    
    #Sub-step 2 - query over index
    
    matching_ids : list[str] = Pinecone_utils.query_pinecone_index(vector)
    
    #Sub-step 3 - get context from wikipedia article
    
    context : str = ""
    
    for index in matching_ids:
        
        context += paragraph_chunks[int(index)]
    
    #Sub-step 4 - update prompt with context
    
    prompts_with_context.update({prompt : context})

#Step 5: Send prompt with context to GPT 3.5 Turbo

responses_file = open(file = os.getenv("TEST_3_TEXT_RESPONSES"), mode = 'a', encoding = 'UTF-8')

for prompt, context in prompts_with_context.items():
    
    prompt_with_context : str = f"""Respond to the following prompt using the context given below it.\nPrompt: {prompt} \nContext: {context}"""

    result : str = OpenAI_utils.get_response(prompt = prompt_with_context, model = "gpt-3.5-turbo", is_question = True)
    
    responses_file.write(f"PROMPT: \n{prompt_with_context}\n RESPONSE: \n{result}\n")
    responses_file.write("---------------------------------------------------------\n")
    

responses_file.close()