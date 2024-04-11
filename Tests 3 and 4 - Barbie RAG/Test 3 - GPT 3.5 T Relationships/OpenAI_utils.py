import os
import openai
import tiktoken
from dotenv import load_dotenv, find_dotenv

#Initailize environment variables
_ = load_dotenv(find_dotenv(filename = "Keys.env"))

client = openai.OpenAI(api_key = os.getenv('OPENAI_API_KEY')) #set client

#Get the number of tokens in a text string.
def num_tokens_from_string(string: str) -> int:
    
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(string)
    num_tokens = len(tokens)
    
    return num_tokens

#Get embedding for a string.
def get_embedding(input_str: str) -> list[float]:

    response = client.embeddings.create(
        input = input_str,
        dimensions = 1536,
        model="text-embedding-3-small"
    )
    
    return response.data[0].embedding

#Prompt ChatGPT for nodes and relationships    
def get_nodes_and_relationships_from_chunk(prompt, model="gpt-3.5-turbo", temperature=0) -> str: #NOTE: TEST 3 WITH GPT3.5 THEN TEST 3.5 WITH GPT4 TURBO PREVIEW
    
    system_message :str = """You are an assistant helping create a knowledge graph about a movie in neo4j.
    Given a paragraph about the movie, you will state all the possible nodes, a label for each node and all relationships possible between each node.
    Each node can only have 1 label.
    Each label must be at least 1 word and at most 5 words.
    You can only give one relationship per line in your responses.
    You can give up to 300 relationships total per response.
    Each relationship must be at least one word and at most 5 words. 
    Do not repeat relationships that are the same or similar.
    You must use underscores in place of spaces in your responses.
    Do not use any special characters other than underscores in your responses.
    
    If you cannot form any relationships then simply state NONE.
    
    Do not number your responses.
    
    Each line in your response should follow the following format:
    
    NodeName(NodeLabel)->Relationship->NodeName(NodeLabel)
    
    Example:
    
    prompt = 
    
    Barbie is a 2023 fantasy comedy film directed by Greta Gerwig from a screenplay she wrote with Noah Baumbach.
    
    your response = 
    
    Barbie(Movie)->is_genre->Fantasy(Genre) 
    Barbie(Movie)->is_genre->Comedy(Genre)
    Barbie(Movie)->released_in->2023(Year)
    Barbie(Movie)->directed_by->Greta_Gerwig(Director)
    Barbie(Movie)->screenplay_written_by->Greta_Gerwig(Director)
    Barbie(Movie)->screenplay_written_by->Noah_Baumbach(Screenwriter)
    Greta_Gerwig(Director)->wrote_screenplay_for_Barbie_with->Noah_Baumbach(Screenwriter)
    """
    
    messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': prompt} 
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature 
    )
    return response.choices[0].message.content