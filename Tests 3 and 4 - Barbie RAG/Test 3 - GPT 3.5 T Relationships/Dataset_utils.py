from OpenAI_utils import num_tokens_from_string

#get wiki article as paragraphs
def get_paragraphs_from_file(article_path: str) -> list[str]:

    with open(article_path, "r", encoding = 'UTF-8') as file:
        
        return file.read().split("\n\n")


def token_chopper(context : str):
    
    while(num_tokens_from_string(context) >= 10000): #16384 is max token limit for gpt 3.5 turbo (prompt + completion) 
        
        # Split the paragraph into lines
        
        lines = context.splitlines()

        # Remove the last line
        lines_without_last = lines[:-1]

        # Join the lines back together
        context = '\n'.join(lines_without_last)
        
    return context