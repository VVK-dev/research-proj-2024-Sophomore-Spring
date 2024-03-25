from langchain_community.graphs import Neo4jGraph

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
        MATCH (contextNode)-[relationship]-(neighbor)
        RETURN contextNode, relationship, neighbor
        """, 
        params= {"openAiApiKey": OpenAIKey,
                "prompt": "Andreas Boba",
                "embeddingModel" : "text-embedding-3-small",
                "top_k": 2}
    )

    #Form clean string of all context nodes and relationships together

    context_string : str = ""
    
    for relation in result:
        
        context_string += f"{relation['relationship'][0]['name']} {relation['relationship'][1]} {relation['relationship'][2]['name']}.\n"

    return context_string