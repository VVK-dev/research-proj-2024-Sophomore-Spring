import os
from openai import OpenAI
import tiktoken

client = OpenAI(os.getenv('OPENAI_API_KEY')) #set client

#Get the number of tokens in a text string.
def num_tokens_from_string(string: str, encoding_name: str = tiktoken.encoding_for_model("text-embedding-3-small")) -> int:
    
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    
    return num_tokens
    
