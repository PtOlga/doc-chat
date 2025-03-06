import os
import time
import sys
import json
import traceback
import warnings
from datetime import datetime
from typing import Optional, List, Dict
import logging

# Настройка логгера
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, BSHTMLLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_core.callbacks import CallbackManager
from langchain_core.documents import Document

# Ignore SSL warnings
warnings.filterwarnings('ignore')

# Initialize environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Status Law Assistant API")

# Конфигурация базы знаний
KB_CONFIG_PATH = "vector_store/kb_config.json"

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
VECTOR_STORE_PATH = "vector_store"
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

# Check write permissions
try:
    if not os.path.exists(VECTOR_STORE_PATH):
        os.makedirs(VECTOR_STORE_PATH)
    test_file_path = os.path.join(VECTOR_STORE_PATH, 'test_write.txt')
    with open(test_file_path, 'w') as f:
        f.write('test')
    os.remove(test_file_path)
    print(f"Write permissions OK for {VECTOR_STORE_PATH}")
except Exception as e:
    print(f"WARNING: No write permissions for {VECTOR_STORE_PATH}: {str(e)}")
    print("Current working directory:", os.getcwd())
    print("User:", os.getenv('USER'))
    sys.exit(1)

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
    try:
        callback_handler = CustomCallbackHandler()
        callback_manager = CallbackManager([callback_handler])
        
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.6,
            api_key=os.getenv("GROQ_API_KEY"),
            callback_manager=callback_manager
        )
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large-instruct"
        )
        return llm, embeddings
    except Exception as e:
        raise Exception(f"Model initialization failed: {str(e)}")

def check_url_availability(url: str) -> bool:
    try:
        response = requests.get(url, verify=False, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Error checking {url}: {str(e)}")
        return False

def load_url_content(url: str) -> List[Document]:
    try:
        response = requests.get(url, verify=False, timeout=30)
        if response.status_code != 200:
            print(f"Failed to load {url}, status code: {response.status_code}")
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return [Document(page_content=text, metadata={"source": url})]
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return []

def build_knowledge_base(embeddings):
    try:
        logger.info("Starting knowledge base construction...")
        kb_config = get_kb_config()
        documents = []
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        
        # Определяем URL для обработки
        urls_to_process = [url for url in URLS if url not in kb_config["processed_urls"]]
        
        if not urls_to_process:
            logger.info("No new URLs to process")
            return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            
        logger.info(f"Processing {len(urls_to_process)} new URLs")
        
        available_urls = [url for url in urls_to_process if check_url_availability(url)]
        logger.info(f"Accessible URLs: {len(available_urls)} out of {len(urls_to_process)}")
        
        for url in available_urls:
            try:
                logger.info(f"Processing {url}")
                docs = load_url_content(url)
                if docs:
                    documents.extend(docs)
                    kb_config["processed_urls"].append(url)
                    logger.info(f"Successfully loaded content from {url}")
                else:
                    logger.warning(f"No content extracted from {url}")
            except Exception as e:
                logger.error(f"Failed to process {url}: {str(e)}")
                continue

        if not documents:
            if kb_config["processed_urls"]:
                logger.info("No new documents to add, loading existing vector store")
                return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            raise Exception("No documents were successfully loaded!")

        logger.info(f"Total new documents loaded: {len(documents)}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50
        )
        logger.info("Splitting documents into chunks...")
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Если есть существующая база знаний, добавляем к ней
        if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
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
try:
    llm, embeddings = init_models()
    vector_store = None

    if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        try:
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Successfully loaded existing knowledge base")
        except Exception as e:
            logger.warning(f"Could not load existing knowledge base, will create new one: {str(e)}")
            vector_store = None
    else:
        logger.info("No existing knowledge base found, will create new one")

    if vector_store is None:
        logger.info("Building new knowledge base...")
        vector_store = build_knowledge_base(embeddings)
        logger.info("Knowledge base built successfully")

except Exception as e:
    logger.error(f"Critical initialization error: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# API endpoints
# API endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Retrieve context
        context_docs = vector_store.similarity_search(request.message)
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
async def rebuild_knowledge_base():
    try:
        global vector_store
        vector_store = build_knowledge_base(embeddings)
        return {"status": "success", "message": "Knowledge base rebuilt successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kb-status")
async def get_kb_status():
    """Get current knowledge base status"""
    kb_config = get_kb_config()
    return {
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
