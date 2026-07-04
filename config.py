import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("PINECONE_API_KEY Loaded:", bool(PINECONE_API_KEY))
print("PINECONE_INDEX:", PINECONE_INDEX)
print("GROQ_API_KEY Loaded:", bool(GROQ_API_KEY))