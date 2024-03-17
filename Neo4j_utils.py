import os
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from langchain_core import documents
from neo4j import Driver, GraphDatabase, EagerResult

URI = os.getenv("NEO4J_URI")
Username, Password = os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")

vector_index : Neo4jVector = Neo4jVector(url = URI, username = Username, password = Password, database = "neo4j").from_existing_index(index_name = "")

driver : Driver = GraphDatabase.driver(URI, auth = (Username, Password))

def check_connectivity():
    
    driver.verify_connectivity()

def get_similar_nodes_from_neo4j_vector_index(prompt : str):
    
    #get the 2 most similar nodes to the given prompt
    
    nodes : list[documents.Document] = vector_index.similarity_search(query = prompt, k = 2)
    
    nodes_with_context : dict = {}
    
    for node in nodes:
        
        nodes_with_context.update({node, None})
        
        
