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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize environment variables
load_dotenv()

app = FastAPI(title="Status Law Assistant API")

# --------------- Model Initialization ---------------
def init_models():
    """Initialize AI models"""
    try:
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.6,
            api_key=os.getenv("GROQ_API_KEY")
        )
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large-instruct"
        )
        return llm, embeddings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")

# --------------- Knowledge Base Management ---------------
VECTOR_STORE_PATH = "vector_store"
URLS = [
    "https://status.law",
    "https://status.law/about",
    "https://status.law/careers",  
    "https://status.law/tariffs-for-services-of-protection-against-extradition",
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
        
        # Log interaction
        log_interaction(request.message, response, context_text)
        
        return ChatResponse(response=response)
            
    except Exception as e:
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
