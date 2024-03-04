from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key="YOUR_API_KEY")

pc.create_index(
  
  name="starter-index",
  dimension=1536,
  metric="cosine",
  spec=PodSpec(
    environment="gcp-starter"
  )
  
)