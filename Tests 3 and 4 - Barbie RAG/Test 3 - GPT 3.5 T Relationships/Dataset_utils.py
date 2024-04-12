import OpenAI_utils

#get wiki article as paragraphs
def get_paragraphs_from_file(article_path: str) -> list[str]:

    with open(article_path, "r", encoding = 'UTF-8') as file:
        
        return file.read().split("\n\n")


def token_chopper(context : str):
    
    while(OpenAI_utils.num_tokens_from_string(context) >= 100000): #128000 is max token limit for gpt 4 turbo (prompt + completion) 
        
        # Split the paragraph into lines
        
        lines = context.splitlines()

        # Remove the last line
        lines_without_last = lines[:-1]

        # Join the lines back together
        context = '\n'.join(lines_without_last)
        
    return context