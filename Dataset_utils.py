from OpenAI_utils import num_tokens_from_string
import csv

#get article names from articles.tsv

def get_data_from_file(articles_path: str) -> list[str]:

    with open(articles_path, "r") as file:
        
        tsv_reader = csv.reader(file, delimiter = '\t')
        
        article_names : list[str] = []
        
        for row in tsv_reader:
            
            if((len(row) == 0) or (row[0].startswith('#'))): #Ignore empty lines and comments 
                continue
            
            article_names.append(row[0])
            
        #the filetext variable now contains a list of chunks
        
        return article_names
    
    
def token_chopper(context : str):
    
    while(num_tokens_from_string(context) >= 2300):
        
        # Split the paragraph into lines
        
        lines = context.splitlines()

        # Remove the last line
        lines_without_last = lines[:-1]

        # Join the lines back together
        context = '\n'.join(lines_without_last)
        
    return context