from OpenAI_utils import num_tokens_from_string

#get wiki article as paragraphs
def get_paragraphs_from_file(article_path: str) -> list[str]:

    with open(article_path, "r") as file:
        
        return file.read().split("\n\n")


def token_chopper(context : str):
    
    while(num_tokens_from_string(context) >= 2000): #~2000 cl100k_base tokens seems to be the max Llama can take as 
                                                    #context
        
        # Split the paragraph into lines
        
        lines = context.splitlines()

        # Remove the last line
        lines_without_last = lines[:-1]

        # Join the lines back together
        context = '\n'.join(lines_without_last)
        
    return context