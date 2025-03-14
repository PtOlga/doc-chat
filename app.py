import os

# Установка переменных окружения для кэша HuggingFace
#os.environ["TRANSFORMERS_CACHE"] = "cache/huggingface"
os.environ["HF_HOME"] = "cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "cache/huggingface"
os.environ["XDG_CACHE_HOME"] = "cache"

# Создание необходимых директорий
os.makedirs("cache/huggingface", exist_ok=True)

import time
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
import json
import traceback
from typing import Dict, List, Optional
from pydantic import BaseModel
from huggingface_hub import Repository, snapshot_download

# Initialize environment variables
load_dotenv()

# Constants for paths and URLs
VECTOR_STORE_PATH = "vector_store"
LOCAL_CHAT_HISTORY_PATH = "chat_history"
DATA_SNAPSHOT_PATH = "data_snapshot"
HF_DATASET_REPO = "Rulga/LS_chat"

URLS = [
    "https://status.law",
    "https://status.law/about",
    "https://status.law/careers",  
    "https://status.law/tariffs-for-services-of-protection-against-extradition",
    "https://status.law/challenging-sanctions",
    "https://status.law/law-firm-contact-legal-protection", 
    "https://status.law/cross-border-banking-legal-issues", 
    "https://status.law/extradition-defense", 
    "https://status.law/international-prosecution-protection", 
    "https://status.law/interpol-red-notice-removal",  
    "https://status.law/practice-areas",  
    "https://status.law/reputation-protection",
    "https://status.law/faq"
]

# Initialize the FastAPI app
app = FastAPI(title="Status Law Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    
class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    
class BuildKnowledgeBaseResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict] = None

# Global variables for models and knowledge base
llm = None
embeddings = None
vector_store = None
kb_info = {
    'build_time': None,
    'size': None,
    'version': '1.1'
}

# --------------- Hugging Face Dataset Integration ---------------
def init_hf_dataset_integration():
    """Initialize integration with Hugging Face dataset for persistence"""
    try:
        # Download the latest snapshot of the dataset if it exists
        if os.getenv("HF_TOKEN"):
            # With authentication if token provided
            snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                local_dir="./data_snapshot",
                token=os.getenv("HF_TOKEN")
            )
        else:
            # Try without authentication for public datasets
            snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                local_dir="./data_snapshot"
            )
        
        # Check if vector store exists in the downloaded data
        if os.path.exists("./data_snapshot/vector_store/index.faiss"):
            # Copy to the local vector store path
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
            os.system(f"cp -r ./data_snapshot/vector_store/* {VECTOR_STORE_PATH}/")
            return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
    
    return False

def upload_to_hf_dataset():
    """Upload the vector store and chat history to the Hugging Face dataset"""
    if not os.getenv("HF_TOKEN"):
        print("HF_TOKEN not set, cannot upload to Hugging Face")
        return False
    
    try:
        # Clone the repository
        repo = Repository(
            local_dir="./data_upload",
            clone_from=HF_DATASET_REPO,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN")
        )
        
        # Copy the vector store files
        if os.path.exists(f"{VECTOR_STORE_PATH}/index.faiss"):
            os.makedirs("./data_upload/vector_store", exist_ok=True)
            os.system(f"cp -r {VECTOR_STORE_PATH}/* ./data_upload/vector_store/")
        
        # Copy the chat history
        if os.path.exists(f"{LOCAL_CHAT_HISTORY_PATH}/chat_logs.json"):
            os.makedirs("./data_upload/chat_history", exist_ok=True)
            os.system(f"cp -r {LOCAL_CHAT_HISTORY_PATH}/* ./data_upload/chat_history/")
        
        # Push to Hugging Face
        repo.push_to_hub(commit_message="Update vector store and chat history")
        return True
    except Exception as e:
        print(f"Error uploading to dataset: {e}")
        return False

# --------------- Enhanced Logging ---------------
def log_interaction(user_input: str, bot_response: str, context: str, conversation_id: str):
    """Log interactions with error handling"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "user_input": user_input,
            "bot_response": bot_response,
            "context": context[:500] if context else "",
            "kb_version": kb_info['version']
        }
        
        os.makedirs(LOCAL_CHAT_HISTORY_PATH, exist_ok=True)
        log_path = os.path.join(LOCAL_CHAT_HISTORY_PATH, "chat_logs.json")
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
        # Upload to Hugging Face after logging
        upload_to_hf_dataset()
            
    except Exception as e:
        print(f"Logging error: {str(e)}")
        print(traceback.format_exc())

# --------------- Model Initialization ---------------
def init_models():
    """Initialize AI models"""
    global llm, embeddings
    
    if not llm:
        try:
            llm = ChatGroq(
                model_name="llama-3.3-70b-versatile",
                temperature=0.6,
                api_key=os.getenv("GROQ_API_KEY")
            )
        except Exception as e:
            print(f"LLM initialization failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"LLM initialization failed: {str(e)}")
    
    if not embeddings:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large-instruct"
            )
        except Exception as e:
            print(f"Embeddings initialization failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Embeddings initialization failed: {str(e)}")
    
    return llm, embeddings

# --------------- Knowledge Base Management ---------------
def build_knowledge_base():
    """Build or update the knowledge base"""
    global vector_store, kb_info
    
    _, _embeddings = init_models()
    
    try:
        start_time = time.time()
        documents = []
        
        # Create folder in advance
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        
        # Load documents
        for url in URLS:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded {url}")
            except Exception as e:
                print(f"Failed to load {url}: {str(e)}")
                continue
                
        if not documents:
            raise HTTPException(status_code=500, detail="No documents loaded!")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, _embeddings)
        vector_store.save_local(
            folder_path=VECTOR_STORE_PATH,
            index_name="index"
        )
        
        # Verify file creation
        if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
            raise HTTPException(status_code=500, detail="FAISS index file not created!")
            
        # Update info
        kb_info.update({
            'build_time': time.time() - start_time,
            'size': sum(
                os.path.getsize(os.path.join(VECTOR_STORE_PATH, f)) 
                for f in ["index.faiss", "index.pkl"]
            ) / (1024 ** 2),
            'version': datetime.now().strftime("%Y%m%d-%H%M%S")
        })
        
        # Upload to Hugging Face
        upload_to_hf_dataset()
        
        return {
            "status": "success",
            "message": "Knowledge base successfully created!",
            "details": kb_info
        }
            
    except Exception as e:
        error_msg = f"Knowledge base creation failed: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

def load_knowledge_base():
    """Load the knowledge base from disk"""
    global vector_store
    
    if vector_store:
        return vector_store
        
    _, _embeddings = init_models()
    
    try:
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            _embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        error_msg = f"Failed to load knowledge base: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return None

# --------------- API Endpoints ---------------
@app.get("/")
async def root():
    """Root endpoint that shows app status"""
    vector_store_exists = os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss"))
    
    return {
        "status": "running",
        "knowledge_base_exists": vector_store_exists,
        "kb_info": kb_info if vector_store_exists else None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/build-kb", response_model=BuildKnowledgeBaseResponse)
async def build_kb_endpoint():
    """Endpoint to build/rebuild the knowledge base"""
    return build_knowledge_base()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Endpoint to chat with the assistant"""
    # Check if knowledge base exists
    if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        raise HTTPException(
            status_code=400, 
            detail="Knowledge base not found. Please build it first with /build-kb"
        )
    
    # Use provided conversation ID or generate a new one
    conversation_id = request.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        # Load models and knowledge base
        _llm, _ = init_models()
        _vector_store = load_knowledge_base()
        
        if not _vector_store:
            raise HTTPException(
                status_code=500, 
                detail="Failed to load knowledge base"
            )
        
        # Retrieve context
        context_docs = _vector_store.similarity_search(request.message)
        context_text = "\n".join([d.page_content for d in context_docs])
        
        # Generate response
        prompt_template = PromptTemplate.from_template('''
            You are a helpful and polite legal assistant at Status Law.
            You answer in the language in which the question was asked.
            Answer the question based on the context provided.
            If you cannot answer based on the context, say so politely and offer to contact Status Law directly via the following channels:
            - For all users: +32465594521 (landline phone).
            - For English and Swedish speakers only: +46728495129 (available on WhatsApp, Telegram, Signal, IMO).
            - Provide a link to the contact form: [Contact Form](https://status.law/law-firm-contact-legal-protection/).
            If the user has questions about specific services and their costs, suggest they visit the page https://status.law/tariffs-for-services-of-protection-against-extradition-and-international-prosecution/ for detailed information.

            Ask the user additional questions to understand which service to recommend and provide an estimated cost. For example, clarify their situation and needs to suggest the most appropriate options.

            Also, offer free consultations if they are available and suitable for the user's request.
            Answer professionally but in a friendly manner.

            Example:
            Q: How can I challenge the sanctions?
            A: To challenge the sanctions, you should consult with our legal team, who specialize in this area. Please contact us directly for detailed advice. You can fill out our contact form here: [Contact Form](https://status.law/law-firm-contact-legal-protection/).

            Context: {context}
            Question: {question}
            
            Response Guidelines:
            1. Answer in the user's language
            2. Cite sources when possible
            3. Offer contact options if unsure
            ''')
        
        chain = prompt_template | _llm | StrOutputParser()
        response = chain.invoke({
            "context": context_text,
            "question": request.message
        })
        
        # Log the interaction
        log_interaction(request.message, response, context_text, conversation_id)
        
        return {
            "response": response,
            "conversation_id": conversation_id
        }
                
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Initialize dataset integration at startup
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    # Try to load existing knowledge base from Hugging Face
    init_hf_dataset_integration()
    
    # Preload embeddings model to reduce first-request latency
    try:
        global embeddings
        if not embeddings:
            embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large-instruct"
            )
    except Exception as e:
        print(f"Warning: Failed to preload embeddings: {e}")

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)