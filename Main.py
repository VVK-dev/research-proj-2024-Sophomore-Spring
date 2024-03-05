import os
from dotenv import load_dotenv, find_dotenv
import Llama2_utils
import OpenAI_utils
import Pinecone_utils
import Dataset_utils

#Initailize global variables
_ = load_dotenv(find_dotenv())

#Step 1: Get chunks from file

filechunks : list[str] = Dataset_utils.get_data_from_file(os.getenv("DATA_FILE_PATH"))

