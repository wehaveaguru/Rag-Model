import chromadb
from main import embeddings,text,text2,text3

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")

collection.add(
    ids= ["id1","id2","id3"],
    documents= [text,text2,text3],
    embeddings=embeddings,
    
    
)

results=collection.query(
    query_texts=["what is an llm"],
    
    n_results=3
)

print(results)
