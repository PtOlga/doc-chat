import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
from typing import Optional, List, Dict
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_core.callbacks import CallbackManager

# Initialize environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Status Law Assistant API")

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
    "https://status.law/law-firm-contact-legal-protection"
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

# Initialize models
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

# Knowledge base management
def build_knowledge_base(embeddings):
    try:
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
            raise Exception("No documents loaded!")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(folder_path=VECTOR_STORE_PATH, index_name="index")
        
        return vector_store
    except Exception as e:
        raise Exception(f"Knowledge base creation failed: {str(e)}")

# Initialize models and knowledge base on startup
llm, embeddings = init_models()
vector_store = None

if os.path.exists(VECTOR_STORE_PATH):
    try:
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Failed to load existing knowledge base: {str(e)}")

if vector_store is None:
    vector_store = build_knowledge_base(embeddings)

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

def log_interaction(user_input: str, bot_response: str, context: str):
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "context": context[:500],
            "kb_version": "1.1"  # You might want to implement version tracking
        }
        
        os.makedirs("chat_history", exist_ok=True)
        with open("chat_history/chat_logs.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
    except Exception as e:
        print(f"Logging error: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)