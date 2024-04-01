import os
import openai
import tiktoken

client = openai.OpenAI(api_key = os.getenv('OPENAI_API_KEY')) #set client

'''
#Get the number of tokens in a text string.
def num_tokens_from_string(string: str, encoding_name: str = tiktoken.encoding_for_model("text-embedding-3-small")) -> int:
    
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    
    return num_tokens
'''
#Get embedding for a string.
def get_embedding(input_str: str) -> list[float]:

    response = client.embeddings.create(
        input = input_str,
        dimensions = 1536,
        model="text-embedding-3-small"
    )
    
    return response.data[0].embedding
    
def get_relation_from_articles(prompt, model="gpt-3.5-turbo-0125", temperature=0) -> str:
    
    system_message :str = """You are an assistant helping create a knowledge graph in neo4j.
    Given 2 entities, you will give the possible relationships ONLY from the first entity to the second entity that would make sense in a knowledge graph.
    DO NOT GIVE RELATIONSHIPS FROM THE SECOND ENTITY TO THE FIRST ENTITY.
    You can give upto 3 relationsips.
    You can only give one relationship per line in your responses.
    Each relationship must be at least one word and at most 5 words. 
    Do not repeat relationships that are semantically similar.
    You must use underscores in place of spaces in your responses.
    Do not use any special characters other than underscores in your responses.
    
    If you cannot form any relationships then simply state NONE.
    
    Do not number or format your response in any way other than given above.
    
    Example:
    
    prompt = 
    Entity 1: 11th Century 
    Entity 2: 12th Century
    Please tell me all the relationships possible from the first entity to the second entity.
    
    your response = 
    precedes
    fedualism_in_europe
    religious_crusades
    growth_of_trade
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