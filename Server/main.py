from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq



# Load environment
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in your .env file.")
os.environ["GROQ_API_KEY"] = groq_api_key

# Flask setup
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": FRONTEND_URL}})

# Load existing vectorstore (Chroma) from disk
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_doc_db", embedding_function=embedding, read_only=True)

@app.route('/')
def index():
    return "Backend is running!"

@app.route('/api/process-url', methods=['POST'])
def process_url():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Initialize LLM
        model = ChatGroq(
            model_name="llama3-8b-8192",
            api_key=groq_api_key,
            temperature=0.6,
            max_tokens=512
        )


        # Prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful assistant that provides concise answers based on the provided documents.

            Context: {context}

            Question: {question}

            Answer:"""
        )

        # Memory
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5,
            output_key="result"
        )

        # Retriever and QA chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        qa = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            memory=memory,
            chain_type_kwargs={"prompt": prompt_template},
            output_key="result"
        )

        # Run query
        result = qa({"query": query})
        
        return jsonify({
            "answer": result["result"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)