import os
import time
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from api import router as analysis_router
from utils import ChatAnalyzer, setup_chat_analysis
import requests.exceptions
import aiohttp
from typing import Union
import uvicorn
import logging
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Базовая настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Определение путей
VECTOR_STORE_PATH = os.path.join(os.getcwd(), "vector_store")
CHAT_HISTORY_PATH = os.path.join(os.getcwd(), "chat_history")

app = FastAPI(title="Status Law Assistant API")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

def check_vector_store():
    """Проверка наличия векторной базы"""
    index_path = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    return os.path.exists(index_path)

@app.get("/")
async def root():
    """Базовый эндпоинт с информацией о состоянии"""
    return {
        "status": "ok",
        "vector_store_ready": check_vector_store(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def get_status():
    """Получение статуса векторной базы"""
    return {
        "vector_store_exists": check_vector_store(),
        "can_chat": check_vector_store(),
        "vector_store_path": VECTOR_STORE_PATH
    }

@app.post("/build-knowledge-base")
async def build_kb():
    """Эндпоинт для построения базы знаний"""
    try:
        if check_vector_store():
            return {
                "status": "exists",
                "message": "Knowledge base already exists"
            }
        
        # Инициализируем embeddings только когда нужно построить базу
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = build_knowledge_base(embeddings)
        
        return {
            "status": "success",
            "message": "Knowledge base built successfully"
        }
    except Exception as e:
        logger.error(f"Failed to build knowledge base: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build knowledge base: {str(e)}"
        )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Эндпоинт чата"""
    if not check_vector_store():
        raise HTTPException(
            status_code=400,
            detail="Knowledge base not found. Please build it first using /build-knowledge-base endpoint"
        )
    
    try:
        # Инициализируем компоненты только при необходимости
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.6,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Остальная логика чата...
        context_docs = vector_store.similarity_search(request.message)
        context_text = "\n".join([d.page_content for d in context_docs])
        
        prompt_template = PromptTemplate.from_template('''
            You are a helpful and polite legal assistant at Status Law.
            Answer the question based on the context provided.
            Context: {context}
            Question: {question}
        ''')
        
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({
            "context": context_text,
            "question": request.message
        })
        
        return ChatResponse(response=response)
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat error: {str(e)}"
        )

# --------------- Knowledge Base Management ---------------
URLS = [
    "https://status.law",
    "https://status.law/about",
    "https://status.law/careers",  
    "https://status.law/tariffs-for-services-against-extradition-en",
    "https://status.law/challenging-sanctions",
    "https://status.law/law-firm-contact-legal-protection"
    "https://status.law/cross-border-banking-legal-issues", 
    "https://status.law/extradition-defense", 
    "https://status.law/international-prosecution-protection", 
    "https://status.law/interpol-red-notice-removal",  
    "https://status.law/practice-areas",  
    "https://status.law/reputation-protection",
    "https://status.law/faq"
]

def build_knowledge_base(_embeddings):
    """Build or update the knowledge base"""
    try:
        start_time = time.time()
        documents = []
        
        # Ensure vector store directory exists
        if not os.path.exists(VECTOR_STORE_PATH):
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        
        for url in URLS:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Failed to load {url}: {str(e)}")
                continue

        if not documents:
            raise HTTPException(status_code=500, detail="No documents loaded")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        vector_store = FAISS.from_documents(chunks, _embeddings)
        vector_store.save_local(
            folder_path=VECTOR_STORE_PATH,
            index_name="index"
        )
        
        if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
            raise HTTPException(status_code=500, detail="FAISS index file not created")
            
        return vector_store
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge base creation failed: {str(e)}")

# --------------- API Models ---------------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# --------------- API Routes ---------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        llm, embeddings = init_models()
        
        if not os.path.exists(VECTOR_STORE_PATH):
            vector_store = build_knowledge_base(embeddings)
        else:
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

        # Add retry logic for network operations
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                context_docs = vector_store.similarity_search(request.message)
                context_text = "\n".join([d.page_content for d in context_docs])
                
                prompt_template = PromptTemplate.from_template('''
                    You are a helpful and polite legal assistant at Status Law.
                    You answer in the language in which the question was asked.
                    Answer the question based on the context provided.
                    
                    # ... остальной текст промпта ...

                    Context: {context}
                    Question: {question}
                    
                    Response Guidelines:
                    1. Answer in the user's language
                    2. Cite sources when possible
                    3. Offer contact options if unsure
                    ''')
                
                chain = prompt_template | llm | StrOutputParser()
                response = chain.invoke({
                    "context": context_text,
                    "question": request.message
                })
                
                log_interaction(request.message, response, context_text)
                return ChatResponse(response=response)
                
            except (requests.exceptions.RequestException, aiohttp.ClientError) as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "error": "Network error after maximum retries",
                            "detail": str(e),
                            "type": "network_error"
                        }
                    )
                await asyncio.sleep(1 * retry_count)  # Exponential backoff
                
    except Exception as e:
        if isinstance(e, (requests.exceptions.RequestException, aiohttp.ClientError)):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Network error occurred",
                    "detail": str(e),
                    "type": "network_error"
                }
            )
        raise HTTPException(status_code=500, detail=str(e))

# --------------- Logging ---------------
def log_interaction(user_input: str, bot_response: str, context: str):
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "context": context[:500],
            "kb_version": datetime.now().strftime("%Y%m%d-%H%M%S")
        }
        
        os.makedirs("chat_history", exist_ok=True)
        log_path = os.path.join("chat_history", "chat_logs.json")
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
    except Exception as e:
        print(f"Logging error: {str(e)}")
        print(traceback.format_exc())

# Add health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Check if models can be initialized
        llm, embeddings = init_models()
        
        # Check if vector store is accessible
        if os.path.exists(VECTOR_STORE_PATH):
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        
        return {
            "status": "healthy",
            "vector_store": "available" if os.path.exists(VECTOR_STORE_PATH) else "not_found"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

# Add diagnostic endpoint
@app.get("/directory-status")
async def check_directory_status():
    """Check status of required directories"""
    return {
        "vector_store": {
            "exists": os.path.exists(VECTOR_STORE_PATH),
            "path": os.path.abspath(VECTOR_STORE_PATH),
            "contents": os.listdir(VECTOR_STORE_PATH) if os.path.exists(VECTOR_STORE_PATH) else []
        },
        "chat_history": {
            "exists": os.path.exists(CHAT_HISTORY_PATH),
            "path": os.path.abspath(CHAT_HISTORY_PATH),
            "contents": os.listdir(CHAT_HISTORY_PATH) if os.path.exists(CHAT_HISTORY_PATH) else []
        }
    }

# Добавим функцию для вывода статуса
def print_startup_status():
    """Print application startup status with rich formatting"""
    try:
        # Create status table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        # Check directories
        vector_store_exists = os.path.exists(VECTOR_STORE_PATH)
        chat_history_exists = os.path.exists(CHAT_HISTORY_PATH)
        
        table.add_row(
            "Vector Store Directory",
            "✅ Created" if vector_store_exists else "❌ Missing"
        )
        table.add_row(
            "Chat History Directory",
            "✅ Created" if chat_history_exists else "❌ Missing"
        )
        
        # Check environment variables
        table.add_row(
            "GROQ API Key",
            "✅ Set" if os.getenv("GROQ_API_KEY") else "❌ Missing"
        )
        
        # Create status panel
        status_panel = Panel(
            table,
            title="[bold blue]Status Law Assistant API Status[/bold blue]",
            border_style="blue"
        )
        
        # Print startup message and status
        console.print("\n")
        console.print("[bold green]🚀 Server started successfully![/bold green]")
        console.print(status_panel)
        console.print("\n[bold yellow]API Documentation:[/bold yellow]")
        console.print("📚 Swagger UI: http://0.0.0.0:8000/docs")
        console.print("📘 ReDoc: http://0.0.0.0:8000/redoc\n")
        
    except Exception as e:
        console.print(f"[bold red]Error printing status: {str(e)}[/bold red]")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="debug"
    )
    
    server = uvicorn.Server(config)
    server.run()
