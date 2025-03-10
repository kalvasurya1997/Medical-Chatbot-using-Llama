from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import pinecone
import os
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers  # Updated import from langchain_community
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import prompt_template  # Import only the prompt template you need

app = Flask(__name__)

# Load environment variables from your .env file
load_dotenv()

# Retrieve Pinecone API key and environment
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')

# Download embeddings using your helper function
embeddings = download_hugging_face_embeddings()

# (Optional) Set these in os.environ if needed by other libraries
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_API_ENV"] = PINECONE_API_ENV

index_name = "medical-chatbot"

# Load your existing Pinecone index using the updated vector store interface
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create a prompt template using your custom prompt from src.prompt
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Instantiate your local LLM using the updated CTransformers class from langchain_community.llms
llm = CTransformers(
    model=r"C:\Users\kalva\AI_Projects\Medical-Chatbot-using-Llama\model\llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        'max_new_tokens': 512,
        'temperature': 0.8,
        'gpu_layers': 0  # Force CPU inference
    }
)

# Test the LLM (optional)
#print(llm("Hello, how are you?"))

# Create a RetrievalQA chain with your LLM and the vector store retriever
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
