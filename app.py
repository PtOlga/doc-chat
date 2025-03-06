import os
import sys
import json
import traceback
import warnings
import asyncio
import aiohttp
from datetime import datetime
from typing import Optional, List, Dict
import logging

# Настройка логгера
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_core.callbacks import CallbackManager
from langchain_core.documents import Document

# Ignore SSL warnings
warnings.filterwarnings('ignore')

# Initialize environment variables
load_dotenv()

# Проверяем наличие необходимых переменных окружения
required_env_vars = ["GROQ_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Проверяем наличие и права доступа к директориям кэша
cache_dir = "/app/.cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    os.chmod(cache_dir, 0o777)

hf_cache_dir = os.path.join(cache_dir, "huggingface")
if not os.path.exists(hf_cache_dir):
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.chmod(hf_cache_dir, 0o777)

logger.info(f"Cache directories initialized: {cache_dir}, {hf_cache_dir}")

# Initialize FastAPI app
app = FastAPI(title="Status Law Assistant API")

# Константы
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
VECTOR_STORE_PATH = "vector_store"
KB_CONFIG_PATH = "vector_store/kb_config.json"
CACHE_DIR = "cache"

# Создаем необходимые директории
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

def get_kb_config():
    if os.path.exists(KB_CONFIG_PATH):
        with open(KB_CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {
        "version": 1,
        "processed_urls": [],
        "last_update": None
    }

def save_kb_config(config):
    os.makedirs(os.path.dirname(KB_CONFIG_PATH), exist_ok=True)
    with open(KB_CONFIG_PATH, 'w') as f:
        json.dump(config, f)

# Models for request/response
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    context: Optional[str] = None

# Global variables
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

# Enhanced logging
class CustomCallbackHandler(ConsoleCallbackHandler):
    def on_chain_end(self, run):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": str(run.id),
            "inputs": run.inputs,
            "outputs": run.outputs,
            "execution_time": run.end_time - run.start_time if run.end_time else None,
            "metadata": run.metadata
        }
        
        os.makedirs("chat_history", exist_ok=True)
        with open("chat_history/detailed_logs.json", "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")

def init_models():
    """Инициализация моделей с обработкой ошибок"""
    try:
        callback_handler = CustomCallbackHandler()
        callback_manager = CallbackManager([callback_handler])
        
        # Инициализация LLM
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.6,
            api_key=os.getenv("GROQ_API_KEY"),
            callback_manager=callback_manager
        )
        
        # Инициализация embeddings с явным указанием кэша
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            cache_folder=hf_cache_dir
        )
        
        logger.info("Models initialized successfully")
        return llm, embeddings
        
    except Exception as e:
        logger.error(f"Model initialization error: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Model initialization failed: {str(e)}")

async def fetch_url(session, url):
    cache_file = os.path.join(CACHE_DIR, f"{url.replace('/', '_').replace(':', '_')}.html")
    
    # Проверяем кэш
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return url, f.read()
    
    try:
        async with session.get(url, ssl=False, timeout=30) as response:
            if response.status == 200:
                content = await response.text()
                # Сохраняем в кэш
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                return url, content
            else:
                logger.warning(f"Failed to load {url}, status code: {response.status}")
                return url, None
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return url, None

def process_html_content(url, html_content):
    if not html_content:
        return None
        
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
        
    # Get text content
    text = soup.get_text()
    
    # Clean up text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    if not text.strip():
        return None
        
    return Document(page_content=text, metadata={"source": url})

async def load_all_urls(urls_to_process):
    documents = []
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls_to_process]
        results = await asyncio.gather(*tasks)
        
    for url, content in results:
        if content:
            doc = process_html_content(url, content)
            if doc:
                documents.append(doc)
                logger.info(f"Successfully processed content from {url}")
            else:
                logger.warning(f"No useful content extracted from {url}")
        else:
            logger.warning(f"Failed to load content from {url}")
            
    return documents

async def build_knowledge_base_async(embeddings, force_rebuild=False):
    """
    Асинхронное построение базы знаний.
    Параметр force_rebuild позволяет принудительно обновить все URL.
    """
    try:
        logger.info("Starting knowledge base construction...")
        kb_config = get_kb_config()
        
        # Определяем URL для обработки
        if force_rebuild:
            urls_to_process = URLS
            kb_config["processed_urls"] = []
            logger.info("Forcing rebuild of entire knowledge base")
        else:
            urls_to_process = [url for url in URLS if url not in kb_config["processed_urls"]]
        
        if not urls_to_process:
            logger.info("No new URLs to process")
            return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            
        logger.info(f"Processing {len(urls_to_process)} new URLs")
        
        documents = await load_all_urls(urls_to_process)

        if not documents:
            if kb_config["processed_urls"] and os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
                logger.info("No new documents to add, loading existing vector store")
                return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            raise Exception("No documents were successfully loaded!")

        logger.info(f"Total new documents loaded: {len(documents)}")
        
        # Увеличиваем размер чанков
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,  # Увеличенный размер чанка
            chunk_overlap=100
        )
        logger.info("Splitting documents into chunks...")
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Если есть существующая база знаний и мы не выполняем полное обновление, добавляем к ней
        if not force_rebuild and os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
            logger.info("Loading existing vector store...")
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            logger.info("Adding new documents to existing vector store...")
            vector_store.add_documents(chunks)
        else:
            logger.info("Creating new vector store...")
            vector_store = FAISS.from_documents(chunks, embeddings)
        
        logger.info("Saving vector store...")
        vector_store.save_local(folder_path=VECTOR_STORE_PATH, index_name="index")
        
        # Обновляем конфигурацию
        for url in urls_to_process:
            if url not in kb_config["processed_urls"]:
                kb_config["processed_urls"].append(url)
            
        kb_config["version"] += 1
        kb_config["last_update"] = datetime.now().isoformat()
        save_kb_config(kb_config)
        
        logger.info(f"Knowledge base updated to version {kb_config['version']}")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error in build_knowledge_base: {str(e)}")
        traceback.print_exc()
        raise Exception(f"Knowledge base creation failed: {str(e)}")

# Initialize models and knowledge base on startup
llm, embeddings = init_models()
vector_store = None

@app.on_event("startup")
async def startup_event():
    global vector_store
    
    # Только загружаем существующую базу при старте, не создаем новую
    if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        try:
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Successfully loaded existing knowledge base")
        except Exception as e:
            logger.warning(f"Could not load existing knowledge base: {str(e)}")
            vector_store = None
    else:
        logger.warning("No existing knowledge base found, please use /rebuild-kb endpoint to create one")

# API endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global vector_store
    
    # Проверяем, инициализирована ли база знаний
    if vector_store is None:
        raise HTTPException(
            status_code=503, 
            detail="Knowledge base not initialized. Please use /rebuild-kb endpoint first."
        )
        
    try:
        # Retrieve context
        context_docs = vector_store.similarity_search(request.message, k=3)  # Ограничиваем количество документов
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
        
        # Log interaction
        log_interaction(request.message, response, context_text)
        
        return ChatResponse(response=response, context=context_text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild-kb")
async def rebuild_knowledge_base(background_tasks: BackgroundTasks, force: bool = False):
    """
    Rebuild knowledge base in the background
    
    - force: если True, перестраивает всю базу знаний с нуля
    """
    global vector_store
    
    try:
        # Запускаем в фоне
        background_tasks.add_task(_rebuild_kb_task, force)
        action = "rebuild" if force else "update"
        return {"status": "success", "message": f"Knowledge base {action} started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _rebuild_kb_task(force: bool = False):
    """Фоновая задача для обновления базы знаний"""
    global vector_store
    try:
        vector_store = await build_knowledge_base_async(embeddings, force_rebuild=force)
        logger.info("Knowledge base rebuild completed successfully")
    except Exception as e:
        logger.error(f"Knowledge base rebuild failed: {str(e)}")

@app.get("/kb-status")
async def get_kb_status():
    """Get current knowledge base status"""
    global vector_store
    
    kb_config = get_kb_config()
    return {
        "initialized": vector_store is not None,
        "version": kb_config["version"],
        "total_urls": len(URLS),
        "processed_urls": len(kb_config["processed_urls"]),
        "pending_urls": len([url for url in URLS if url not in kb_config["processed_urls"]]),
        "last_update": kb_config["last_update"]
    }

def log_interaction(user_input: str, bot_response: str, context: str):
    try:
        kb_config = get_kb_config()
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "context": context[:500],
            "kb_version": kb_config["version"]  # Используем актуальную версию
        }
        
        os.makedirs("chat_history", exist_ok=True)
        with open("chat_history/chat_logs.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
    except Exception as e:
        logger.error(f"Logging error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
