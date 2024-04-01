import os
from dotenv import load_dotenv, find_dotenv
import Neo4j_utils
from langchain_community.graphs import Neo4jGraph

#Initailize global variables
_ = load_dotenv(find_dotenv(filename = "Keys.env"))

#Load Keys

Neo4j_URI = os.getenv("NEO4J_URI")
Username, Password = os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")
OpenAIKey = os.getenv("OPENAI_API_KEY")

#Load Graph

knowledge_graph = Neo4jGraph(url = Neo4j_URI, username = Username, password = Password, database = "neo4j")

#Populate graph

neo4j_query_counter : int = 0

#Neo4j_utils.create_neo4j_nodes(categories_path = os.getenv("CATEGORIES_PATH"), knowledge_graph = knowledge_graph, neo4j_query_counter = neo4j_query_counter)

Neo4j_utils.create_neo4j_relationships(links_path = os.getenv("LINKS_PATH"), knowledge_graph = knowledge_graph, neo4j_query_counter = neo4j_query_counter)