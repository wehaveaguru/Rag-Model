from sentence_transformers import SentenceTransformer


f1=open("01_embeddings_basics.txt","r")
f2=open("02_chunking_strategies.txt","r")
f3=open("03_vector_databases.txt","r")
text=f1.read()
text2=f2.read()
text3=f3.read()
docs=[text,text2,text3]
chunks=''
start=0
window=500
overlap=50
while start<len(text):
    end=start+window
    chunks+=text[start:end]
    start+=window-overlap
chunks2=''
start=0
while start<len(text2):
    end=start+window
    chunks2+=text2[start:end]
    start+=window-overlap
chunks3=''
start=0
while start<len(text3):
    end=start+window
    chunks3+=text3[start]
    start+=window-overlap

 # 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
        chunks,chunks2,chunks3
    ]

#     # 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
#     # [3, 384]

#     # 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
#     # tensor([[1.0000, 0.6660, 0.1046],
#     #         [0.6660, 1.0000, 0.1411],
#     #         [0.1046, 0.1411, 1.0000]])