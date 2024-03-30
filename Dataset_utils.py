from OpenAI_utils import num_tokens_from_string
import csv

#get article names from articles.tsv

def get_data_from_file(articles_path: str) -> list[str]:

    with open(articles_path, "r") as file:
        
        tsv_reader = csv.reader(file, delimiter = '\t')
        
        article_names : list[str] = []
        
        for row in tsv_reader:
            
            article_names.append(row)
            
        #the filetext variable now contains a list of chunks
        
        return article_names
        
        
#Method to calculate Costs

def CalculateCosts(Filechunks : list[str] = None, Filechunk :str = None, isLlama2 : bool = False) -> float:
    
    num_tokens : int = 0
    
    if(Filechunks is None):
        
        num_tokens += num_tokens_from_string(Filechunk)
    
    else:
        
        for chunk in Filechunks:
            
            num_tokens += num_tokens_from_string(string = chunk)
    
    if(isLlama2):
        
        return (num_tokens * 0.0000002)
    
    else:
        
        #If not Llama 2 then text-embedding-3-small
        
        return (num_tokens * 0.00000002)