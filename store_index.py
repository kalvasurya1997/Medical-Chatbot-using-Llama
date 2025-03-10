from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone  # Updated import for vector store
import pinecone
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore

# Load environment variables from your .env file
load_dotenv()

# Retrieve Pinecone API key and environment from environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')

# Process your data
extracted_data = load_pdf(data='C:\\Users\\kalva\\AI_Projects\\Medical-Chatbot-using-Llama\\data')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Define your index name
index_name = "medical-chatbot"

# Initialize Pinecone using the new interface
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PINECONE_API_KEY)

# Optionally, check that your index exists:
existing_indexes = pc.list_indexes().names()
print("Existing indexes:", existing_indexes)
if index_name not in existing_indexes:
    print(f"Warning: Index '{index_name}' not found. Please create it first.")
else:
    print(f"Index '{index_name}' found. Proceeding to use it.")

# Create a Pinecone vector store from your documents.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)

print("Index upsert complete.")
