import os
import time
import json
import traceback
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize environment variables
load_dotenv()

# --------------- Enhanced Logging System ---------------
def log_interaction(user_input: str, bot_response: str, context: str):
    """Log user interactions with context and error handling"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "context": context,
            "model": "llama-3.3-70b-versatile",
            "kb_version": st.session_state.kb_info.get('version', '1.0')
        }
        
        os.makedirs("chat_history", exist_ok=True)
        log_path = os.path.join("chat_history", "chat_logs.json")
        
        # Atomic write operation with UTF-8 encoding
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
    except Exception as e:
        error_msg = f"Logging error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        st.error("Error saving interaction log. Please contact support.")

# --------------- Page Configuration ---------------
st.set_page_config(
    page_title="Status Law Assistant",
    page_icon="⚖️",
    layout="wide",
    menu_items={
        'About': "### Legal AI Assistant powered by Status.Law"
    }
)

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

def init_models():
    """Initialize AI models with caching"""
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.6,
        api_key=os.getenv("GROQ_API_KEY")
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct"
    )
    return llm, embeddings

def build_knowledge_base(embeddings):
    """Create or update the vector knowledge base"""
    start_time = time.time()
    
    documents = []
    with st.status("Building knowledge base..."):
        for url in URLS:
            try:
                loader = WebBaseLoader(url)
                documents.extend(loader.load())
            except Exception as e:
                st.error(f"Failed to load {url}: {str(e)}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    
    # Update version information
    st.session_state.kb_info.update({
        'build_time': time.time() - start_time,
        'size': sum(os.path.getsize(f) for f in os.listdir(VECTOR_STORE_PATH)) / (1024 ** 2),
        'version': datetime.now().strftime("%Y%m%d-%H%M%S")
    })
    
    return vector_store

# --------------- Chat Interface ---------------
def main():
    llm, embeddings = init_models()
    
    # Initialize or load knowledge base
    if not os.path.exists(VECTOR_STORE_PATH):
        if st.button("Initialize Knowledge Base"):
            with st.spinner("Creating knowledge base..."):
                st.session_state.vector_store = build_knowledge_base(embeddings)
                st.rerun()
        return
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
    
    # Display chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # Process user input
    if user_input := st.chat_input("Ask your legal question"):
        # Display user message
        st.chat_message("user").write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question..."):
                try:
                    # Retrieve relevant context
                    context_docs = st.session_state.vector_store.similarity_search(user_input)
                    context_text = "\n".join(d.page_content for d in context_docs)
                    
                    # Generate response
                    prompt_template = PromptTemplate.from_template("""
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
                    """)
                    
                    response = (prompt_template | llm | StrOutputParser()).invoke({
                        "context": context_text,
                        "question": user_input
                    })
                    
                    # Display and log interaction
                    st.write(response)
                    log_interaction(user_input, response, context_text)
                    st.session_state.messages.extend([
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": response}
                    ])
                    
                except Exception as e:
                    error_msg = f"Processing error: {str(e)}\n{traceback.format_exc()}"
                    st.error("Error processing request. Please try again.")
                    print(error_msg)
                    log_interaction(user_input, "SYSTEM_ERROR", context_text)

if __name__ == "__main__":
    main()