import os
from dotenv import load_dotenv, find_dotenv
import Neo4j_utils_LOCAL
from langchain_community.graphs import Neo4jGraph

#Initailize environment variables
_ = load_dotenv(find_dotenv(filename = "Keys.env"))

#Load Keys

Neo4j_URI = os.getenv("NEO4J_URI_LOCAL")
Username, Password = os.getenv("NEO4J_USERNAME_LOCAL"), os.getenv("NEO4J_PASSWORD_LOCAL")
OpenAIKey = os.getenv("OPENAI_API_KEY")

#Load Graph

knowledge_graph = Neo4jGraph(url = Neo4j_URI, username = Username, password = Password, database = "neo4j")

#Populate graph

Neo4j_utils_LOCAL.create_neo4j_nodes(categories_path = os.getenv("CATEGORIES_PATH"), knowledge_graph = knowledge_graph)

Neo4j_utils_LOCAL.create_neo4j_relationships(links_path = os.getenv("LINKS_PATH"), knowledge_graph = knowledge_graph)