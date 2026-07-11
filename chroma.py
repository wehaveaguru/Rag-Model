import chromadb
from main import embeddings

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")

collection.add(
    ids= ["id1","id2"],
    documents= ["This is a document about pineapples",
                "This is a document about apples"],
    embeddings=embeddings,
    metadatas=[
        {
            "subject":"pineapple"
        },
        {
            "subject" : "apple"
        }
    ]
)

results=collection.query(
    query_texts=["This document is about pomegrenates"],
    
    n_results=2
)

print(results)
