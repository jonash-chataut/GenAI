from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

documents=[
    "Kathmandu is the capital of Nepal",
    "Mount Everest is the tallest mountain in the world",
    "Python is a programming language",
    "Football is a popular sport"

]

query='tell me the capital city of Nepal'

doc_embeddings=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index,score=(sorted(list(enumerate(scores)),key=lambda x: x[1])[-1])

print(query)
print(documents[index])
print("similarity score is:",score)

