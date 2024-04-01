from langchain_community.graphs import Neo4jGraph
import urllib.parse
import csv
import OpenAI_utils
import time

#Create knowledge graph in neo4j from categories.tsv
def create_neo4j_nodes(categories_path : str, knowledge_graph : Neo4jGraph, neo4j_query_counter : int):
    
    with open(categories_path, mode="r") as categories:
        
        tsv_reader = csv.reader(categories, delimiter = '\t')
        
        for row in tsv_reader:
            
            if((len(row) == 0) or (row[0].startswith('#'))): #Ignore empty lines and comments 
                continue
            
            #decode name and category of each article
            
            article :list[str] = [urllib.parse.unquote(row[0]).replace("_", " "), urllib.parse.unquote(row[1])] 
            #remove underscores to make text embeddings better
            
            article[1] = article[1].split(".")[-1] #get most specific category to set as label for node
        
            #query to create the node in the graph only if not hitting rate limit
        
            if(neo4j_query_counter >= 125):
                
                #if hitting rate limit, wait 1 minute for rate limit to refresh
                
                time.sleep(60)
                neo4j_query_counter = 0
                 
            #In categories.tsv, multiple categories are given to the same article, but each one is in a different 
            #line. So, if we encounter the same article name, we should add the additional category as a label 
            #to the existing node, rather than creating a new one
            
            knowledge_graph.query("""
            MERGE (
                node:Entity {
                    name: $name
                    })
            ON CREATE SET node:""" + article[1] + "\nON MATCH SET node:" + article[1],   
            params = {"name" : article[0]})
            
            #ON CREATE SET and ON MATCH SET can't take variables, so I'm simply concantenating strings here


#Create relationships between nodes in the knowledge graph by sending them to ChatGPT            
def create_neo4j_relationships(links_path : str, knowledge_graph : Neo4jGraph):
    
    with open(links_path, mode="r",encoding="URL") as categories:
        
        tsv_reader = csv.reader(categories, delimiter = '\t')
        
        for row in tsv_reader:
            
            if(row[0] is '#'): #Ignore comments
                continue
            
            links = urllib.parse.unquote(row).split('\t') #decode names of linked articles
            
            #TODO: TEST AND FIX THIS
            
            prompt : str = f"""Entity 1: {links[0]} 
            Entity 2: {links[1]}
            Please tell me all the relationships possible from the first entity to the second entity."""
            
            relationships = OpenAI_utils.get_relation_from_articles(prompt = prompt)
            
            if("NONE" in relationships):
                continue
            else:
                
                cypher_query : str = """
                    MATCH (
                        nodeA:Entity
                        {
                        name: $name1
                        }),
                        (
                        nodeB:Entity 
                        {
                        name: $name2
                        })
                    MERGE (nodeA)-[r:$relationship]->(nodeB)
                    """
                
                for relation in relationships.splitlines():
                    
                    knowledge_graph.query(
                        cypher_query, 
                        params = { "name1" : links[0], "name2" : links[1], "relationship" : relation})


#Create vector index if it doesn't already exist
def create_neo4j_vector_index(knowledge_graph : Neo4jGraph):
    
    #excecute a cypher query to create a vector index suited for OpenAI's text-embedding-3-small
    
    knowledge_graph.query("""
    CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
    FOR (n:Entity) ON (n.nameEmbedding) 
    OPTIONS { indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }}"""
    )

    #All nodes have the 'Entity' label, so this vector index will be for all nodes in the graph


#Populate vector index
def populate_neo4j_vector_index(knowledge_graph : Neo4jGraph, OpenAIKey : str):
    
    #execute a cypher query to populate the vector index with vectors for all nodes
    
    knowledge_graph.query("""
        MATCH (n) WHERE n.name IS NOT NULL AND n.nameEmbedding IS NULL
        WITH n, genai.vector.encode(
            n.name, 
            "OpenAI", 
            {
            token: $openAiApiKey,
            model: $embeddingModel
            }) AS vector
        CALL db.create.setNodeVectorProperty(n, "nameEmbedding", vector)
        """, 
        params={"openAiApiKey": OpenAIKey, "embeddingModel": "text-embedding-3-small"} )


#Search vector index to get top 2 nodes related to prompt and all of their relationships and neighbors 
def search_neo4j_vector_index(knowledge_graph : Neo4jGraph, OpenAIKey : str, prompt : str) -> str:
    
    #execute a cypher query to get 2 most similar nodes to prompt and all of their relationships and neighbors
    
    result = knowledge_graph.query("""
        WITH genai.vector.encode(
            $prompt, 
            "OpenAI", 
            {
            token: $openAiApiKey,
            model: $embeddingModel
            }) AS prompt_embedding
        CALL db.index.vector.queryNodes(
            'entity_embeddings', 
            $top_k, 
            prompt_embedding
            ) YIELD node AS contextNode
        MATCH (neighbor)-[relationship]-(contextNode)
        RETURN contextNode, relationship, neighbor
        """, 
        params= {"openAiApiKey": OpenAIKey,
                "prompt": prompt,
                "embeddingModel" : "text-embedding-3-small",
                "top_k": 2}
    )

    #Form clean string of all context nodes and relationships together

    context_string : str = ""
    
    for relation in result:
        
        context_string += f"{relation['relationship'][0]['name']} {relation['relationship'][1]} {relation['relationship'][2]['name']}.\n"

    return context_string