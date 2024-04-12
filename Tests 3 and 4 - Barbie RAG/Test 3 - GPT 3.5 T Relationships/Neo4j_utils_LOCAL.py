from langchain_community.graphs import Neo4jGraph
import OpenAI_utils
from typing import List, Dict, Any
import time

#---HELPER METHODS TO UTILITY METHODS---#

#In case chatgpt still used any special characters by mistake
def format_cgpt_response_for_cypher(relationship_piece : str) -> str:
    
    relationship_piece = relationship_piece.strip().replace(' ','_').replace("'","").replace(".","_").replace("-","_").replace('__','_')

    return relationship_piece


#Given a node formatted as NodeName(Label), extract the name and label of the node
def extract_name_and_label_from_string(text : str) -> tuple[str, str]:
    
    text_pieces = text.split('(')
    
    text_pieces[0] = format_cgpt_response_for_cypher(text_pieces[0])
    text_pieces[1] = format_cgpt_response_for_cypher(text_pieces[1])
    
    name_label : tuple[str, str] = [text_pieces[0] , text_pieces[1].strip(')')] #name, label

    return name_label

#Method to neatly format results given from querying the neo4j knowledge graph 
def format_result(result: List[Dict[str, Any]]) -> str:
    
    #Form clean string of all nodes and relationships together

    result_as_formatted_string : str = ""
    
    for relation in result:
        
        nodeA_labels : list = relation['nodeA_labels']
        nodeA_label = nodeA_labels[1] #label of first nodes
        
        nodeB_labels : list = relation['nodeB_labels']
        nodeB_label = nodeB_labels[1] #label of second node
        
        result_as_formatted_string += f"{relation['relationship'][0]['name']}({nodeA_label})->{relation['relationship'][1]}->{relation['relationship'][2]['name']}({nodeB_label}).\n"
    
    return result_as_formatted_string
        

#---UTLITY METHODS---#

#Create nodes and relationships between them in the knowledge graph by sending chunks to ChatGPT            
def populate_neo4j_graph(chunks : list[str], knowledge_graph : Neo4jGraph):
    
    chatgpt_prompt_counter : int = 0
    
    for chunk in chunks:
        
        #get nodes and relationships currently present in graph
        
        cypher_query1 = """
        MATCH (nodeA)-[relationship]-(nodeB)
        RETURN nodeA, labels(nodeA) AS nodeA_labels, relationship, nodeB, labels(nodeB) AS nodeB_labels
        """
        
        result = knowledge_graph.query(query = cypher_query1)
        
        graph : str = format_result(result = result)
        
        #get nodes and relationships by prompting chatgpt
        
        chatgpt_prompt : str = f"""
        Help me create a knowledge graph about the Barbie movie.
        
        Below is a paragraph from an article about the Barbie movie, delimited by ||. Using the paragraph, please tell me all the possible nodes and relationships that can be created in a knowledge graph about the movie.
        
        Paragraph:
        
        ||{chunk}||
        
        Here are the nodes and relationships in the graph so far:
        
        {graph}
        """
        
        if(chatgpt_prompt_counter >= 500):
            
            #if rate limit hit, wait 1 minute to refresh it 
            
            time.sleep(60)
            chatgpt_prompt_counter = 0
            
        cgpt_graph_entities = OpenAI_utils.get_nodes_and_relationships_from_chunk(prompt = chatgpt_prompt)
        
        for relation in cgpt_graph_entities.splitlines():
            
            relation_pieces = relation.split('->')
            
            FirstNode = extract_name_and_label_from_string(relation_pieces[0])
            
            #Special cases---
            
            if(relation_pieces[1] == "caused_Barbie's_existential_crisis(Action)  "):
                relation_pieces[1] = "caused_existential_crisis_for"
                relation_pieces.append("Stereotypical_Barbie(Character)")
            
            #---
            
            SecondNode = extract_name_and_label_from_string(relation_pieces[2])
            #relation_pieces[1] will be the relationship between the 2 nodes, no need to format it
            
            #reformat all pieces of the relationship so that it works in a cypher query
            
            FirstNode[0] = format_cgpt_response_for_cypher(FirstNode[0])
            FirstNode[1] = format_cgpt_response_for_cypher(FirstNode[1])
            SecondNode[0] = format_cgpt_response_for_cypher(SecondNode[0])
            SecondNode[1] = format_cgpt_response_for_cypher(SecondNode[1])
            
            relation_pieces[1] = format_cgpt_response_for_cypher(relation_pieces[1])
            
            #cypher query to create relationship between 2 nodes; if either node doesnt exist it will create them
            cypher_query2 = f"MERGE (nodeA:{FirstNode[1]}:Entity" + " {name: $name1})\n"
            cypher_query2 += f"MERGE (nodeB:{SecondNode[1]}:Entity" + " {name: $name2})\n"
            cypher_query2 += f"MERGE (nodeA)-[r:{relation_pieces[1]}]->(nodeB)"
            
            knowledge_graph.query(query = cypher_query2, params = {"name1" : FirstNode[0], "name2" : SecondNode[0]})            
                            

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
        MATCH (n) WHERE n.name IS NOT NULL
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
        MATCH (contextNode)-[relationship]-(neighbor)
        RETURN contextNode, relationship, neighbor
        """, 
        params= {"openAiApiKey": OpenAIKey,
                "prompt": prompt,
                "embeddingModel" : "text-embedding-3-small",
                "top_k": 1}
    )



    return format_result(result=result)