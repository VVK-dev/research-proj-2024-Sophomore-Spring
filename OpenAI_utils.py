import os
import openai
import tiktoken

client = openai.OpenAI(os.getenv('OPENAI_API_KEY')) #set client

#Get the number of tokens in a text string.
def num_tokens_from_string(string: str, encoding_name: str = tiktoken.encoding_for_model("text-embedding-3-small")) -> int:
    
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    
    return num_tokens
    
#Get embedding for a string.
def get_embedding(input_str: str) -> list[float]:

    response = client.embeddings.create(
        input = input_str,
        dimensions = 1536,
        model="text-embedding-3-small"
    )
    
    return response.data[0].embedding
    
def get_relation_from_articles(prompt, model="gpt-3.5-turbo", temperature=0) -> str:
    
    system_message :str = """You are an assistant helping create a knowledge graph in neo4j.
    Given 2 entities, you will give the possible relationships from the first to the second that would make sense in a knowledge graph.
    You can only give one relationship per line in your responses.
    Each relationship must be at least one word and at most 3 words. 
    You must use underscores in place of spaces in your responses.
    
    
    
    If no relationships can be made between the 2 entities given, respond only with the word delimited by //.
    Do not include the delimiter in your response.
    
    //NONE//  
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