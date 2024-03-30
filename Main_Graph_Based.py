import os
from dotenv import load_dotenv, find_dotenv
import Llama2_utils
import Neo4j_utils
from langchain_community.graphs import Neo4jGraph
from Dataset_utils import CalculateCosts
import sys
import time

#Initailize global variables
_ = load_dotenv(find_dotenv(filename = "Keys.env"))

#Load Keys

Neo4j_URI = os.getenv("NEO4J_URI")
Username, Password = os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")
OpenAIKey = os.getenv("OPENAI_API_KEY")

#Load Graph

knowledge_graph = Neo4jGraph(url = Neo4j_URI, username = Username, password = Password, database = "neo4j")

#Step 1: Create vector index if it doesn't already exist

Neo4j_utils.create_neo4j_vector_index(knowledge_graph)
