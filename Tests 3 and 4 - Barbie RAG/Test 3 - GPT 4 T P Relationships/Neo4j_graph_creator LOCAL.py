import os
from dotenv import load_dotenv, find_dotenv
import Neo4j_utils_LOCAL
from langchain_community.graphs import Neo4jGraph
from Dataset_utils import get_paragraphs_from_file

#Initailize environment variables
_ = load_dotenv(find_dotenv(filename = "Keys.env"))

#Load Keys

Neo4j_URI = os.getenv("NEO4J_URI_LOCAL")
Username, Password = os.getenv("NEO4J_USERNAME_LOCAL"), os.getenv("NEO4J_PASSWORD_LOCAL")
OpenAIKey = os.getenv("OPENAI_API_KEY")

BarbiePath = os.getenv("BARBIE_ARTICLE_GRAPH_PATH")

#Load Graph

knowledge_graph = Neo4jGraph(url = Neo4j_URI, username = Username, password = Password, database = "neo4j")

#Populate graph


Neo4j_utils_LOCAL.populate_neo4j_graph(knowledge_graph = knowledge_graph, chunks = get_paragraphs_from_file(article_path = BarbiePath))