import os
import time
import streamlit as st
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

# Initialize environment variables
load_dotenv()

# --------------- Session State Initialization ---------------
def init_session_state():
    """Initialize all required session state variables"""
    defaults = {
        'kb_info': {
            'build_time': None,
            'size': None,
            'version': '1.1'
        },
        'messages': [],
        'vector_store': None,
        'models_initialized': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --------------- Enhanced Logging ---------------
def log_interaction(user_input: str, bot_response: str, context: str):
    """Log interactions with error handling"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "context": context[:500],  # Store first 500 chars of context
            "kb_version": st.session_state.kb_info['version']
        }
        
        os.makedirs("chat_history", exist_ok=True)
        log_path = os.path.join("chat_history", "chat_logs.json")
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
    except Exception as e:
        st.error(f"Logging error: {str(e)}")
        print(traceback.format_exc())

# --------------- Model Initialization ---------------
@st.cache_resource
def init_models():
    """Initialize AI models with caching"""
    try:
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.6,
            api_key=os.getenv("GROQ_API_KEY")
        )
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large-instruct"
        )
        st.session_state.models_initialized = True
        return llm, embeddings
    except Exception as e:
        st.error(f"Model initialization failed: {str(e)}")
        st.stop()

# --------------- Knowledge Base Management ---------------
VECTOR_STORE_PATH = "vector_store"
URLS = [
    "https://status.law",
    "https://status.law/about",
    # ... other URLs ...
]

def build_knowledge_base(_embeddings):
    """Build or update the knowledge base"""
    try:
        start_time = time.time()
        documents = []
        
        with st.status("Building knowledge base..."):
            # Создаем папку заранее
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
            
            # Загрузка документов
            for url in URLS:
                try:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    documents.extend(docs)
                    st.write(f"✓ Loaded {url}")
                except Exception as e:
                    st.error(f"Failed to load {url}: {str(e)}")
                    continue  # Продолжаем при ошибках загрузки

            if not documents:
                st.error("No documents loaded!")
                return None

            # Разделение на чанки
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(documents)
            
            # Явное сохранение
            vector_store = FAISS.from_documents(chunks, _embeddings)
            vector_store.save_local(
                folder_path=VECTOR_STORE_PATH,
                index_name="index"
            )
            
            # Проверка создания файлов
            if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
                raise RuntimeError("FAISS index file not created!")
                
            # Обновление информации
            st.session_state.kb_info.update({
                'build_time': time.time() - start_time,
                'size': sum(
                    os.path.getsize(os.path.join(VECTOR_STORE_PATH, f)) 
                    for f in ["index.faiss", "index.pkl"]
                ) / (1024 ** 2),
                'version': datetime.now().strftime("%Y%m%d-%H%M%S")
            })
            
            st.success("Knowledge base successfully created!")
            return vector_store
            
    except Exception as e:
        st.error(f"Knowledge base creation failed: {str(e)}")
        # Отладочная информация
        st.write("Debug info:")
        st.write(f"Documents loaded: {len(documents)}")
        st.write(f"Chunks created: {len(chunks) if 'chunks' in locals() else 0}")
        st.write(f"Vector store path exists: {os.path.exists(VECTOR_STORE_PATH)}")
        st.stop()
# --------------- Main Application ---------------
def main():
    # Initialize session state first
    init_session_state()
    
    # Page configuration
    st.set_page_config(
        page_title="Status Law Assistant",
        page_icon="⚖️",
        layout="wide"
    )
    
    # Display header
    st.markdown('''
        <h1 style="border-bottom: 2px solid #444; padding-bottom: 10px;">
            ⚖️ <a href="https://status.law/" style="text-decoration: none; color: #2B5876;">Status.Law</a> Legal Assistant
        </h1>
    ''', unsafe_allow_html=True)

    # Initialize models
    llm, embeddings = init_models()
    
    # Knowledge base initialization
    if not os.path.exists(VECTOR_STORE_PATH):
        st.warning("Knowledge base not initialized")
        if st.button("Create Knowledge Base"):
            st.session_state.vector_store = build_knowledge_base(embeddings)
            st.rerun()
        return
    
    if not st.session_state.vector_store:
        try:
            st.session_state.vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Failed to load knowledge base: {str(e)}")
            st.stop()

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your legal question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Retrieve context
                context_docs = st.session_state.vector_store.similarity_search(prompt)
                context_text = "\n".join([d.page_content for d in context_docs])
                
                # Generate response
                prompt_template = PromptTemplate.from_template('''
                You are a helpful and polite legal assistant at Status.Law.
                Answer in the language in which the question was asked.
                Answer the question based on the context provided.

                Context: {context}
                Question: {question}
                ''')
                
                chain = prompt_template | llm | StrOutputParser()
                response = chain.invoke({
                    "context": context_text,
                    "question": prompt
                })
                
                # Display and log
                st.markdown(response)
                log_interaction(prompt, response, context_text)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                log_interaction(prompt, error_msg, "")
                print(traceback.format_exc())

if __name__ == "__main__":
    main()