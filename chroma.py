import chromadb
from groq import Groq
import os
from dotenv import load_dotenv
from main import embeddings,text,text2,text3

load_dotenv()

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")

api_key=os.getenv("token")
if api_key:
    print("api key retrieval successful")
else:
    print("failed")

collection.add(
    ids= ["id1","id2","id3"],
    documents= [text,text2,text3],
    embeddings=embeddings,
    
    
)

results=collection.query(
    query_texts=["what is an llm"],
    
    n_results=3
)

rd=results['documents'][0]

client=Groq(
    api_key=api_key
)
production_prompt = f"""You are a precise, factual assistant. Your task is to answer the user's question using ONLY the provided context blocks.

[CRITICAL CONSTRAINTS]
- Base your answer strictly on the text provided below. Do not assume or extrapolate.
- If the context doesn't contain the answer, reply exactly with: "I cannot find the answer in the provided documents."

[START OF CONTEXT]
{rd}
[END OF CONTEXT]

User Question: {'How do embeddings help vector databases'}
Helpful Answer:"""
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": production_prompt,
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)