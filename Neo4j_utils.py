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
