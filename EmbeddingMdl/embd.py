from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# model choice
# "all-MiniLM-L6-v2" is a good open-source embedding model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about kohli"

# Generate embeddings
doc_embeddings = embed_model.embed_documents(documents)
query_embedding = embed_model.embed_query(query)

# Compute similarity with cosine
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Pick highest score
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)
