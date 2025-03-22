import os
import sys
import threading
import time
import requests
from requests.exceptions import ConnectionError

# Установка конфигурационной директории для Matplotlib
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), 'cache', 'matplotlib')
# Создаем директорию с правильными правами
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Add project root to Python path
current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our main application
from api.fastapi_server import app as fastapi_app

def wait_for_fastapi(timeout=30, interval=0.5):
    """Wait for FastAPI server to start"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://127.0.0.1:8000/health")
            if response.status_code == 200:
                print("FastAPI server is ready!")
                return True
        except ConnectionError:
            time.sleep(interval)
    return False

# Run FastAPI server in a separate thread
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

# Start FastAPI in a background thread
fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
fastapi_thread.start()

# Wait for FastAPI to start
if not wait_for_fastapi():
    print("Failed to start FastAPI server")
    sys.exit(1)

# Create a Gradio interface that will proxy requests to FastAPI
def chat_with_api(message, conversation_id=None):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"message": message, "conversation_id": conversation_id}
        )
        if response.status_code == 200:
            data = response.json()
            return data["response"], data["conversation_id"]
        else:
            return f"Error: {response.status_code} - {response.text}", conversation_id
    except Exception as e:
        return f"API connection error: {str(e)}", conversation_id

def build_kb():
    try:
        response = requests.post("http://127.0.0.1:8000/build-kb")
        if response.status_code == 200:
            return f"Success: {response.json()['message']}"
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"API connection error: {str(e)}"

def get_kb_size():
    """Calculate actual knowledge base size from files"""
    vector_store_path = "vector_store"
    try:
        total_size = 0
        for file in ["index.faiss", "index.pkl"]:
            file_path = os.path.join(vector_store_path, file)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
        return total_size / (1024 * 1024)  # Convert to MB
    except Exception:
        return None

# Добавим функцию проверки статуса базы знаний
def check_kb_status():
    try:
        response = requests.get("http://127.0.0.1:8000/")
        if response.status_code == 200:
            data = response.json()
            if data["knowledge_base_exists"]:
                kb_info = data["kb_info"]
                version = kb_info.get('version', 'N/A')
                
                # Получаем реальный размер файлов
                actual_size = get_kb_size()
                size_text = f"{actual_size:.2f} MB" if actual_size is not None else "N/A"
                
                return f"✅ База знаний готова к работе\nВерсия: {version}\nРазмер: {size_text}"
            else:
                # Проверяем, есть ли файлы на диске
                if os.path.exists(os.path.join("vector_store", "index.faiss")):
                    actual_size = get_kb_size()
                    size_text = f"{actual_size:.2f} MB" if actual_size is not None else "N/A"
                    return f"✅ База знаний найдена на диске\nРазмер: {size_text}\nТребуется перезагрузка сервера"
                return "❌ База знаний не создана. Нажмите кнопку 'Create/Update Knowledge Base'"
    except Exception as e:
        # Проверяем наличие файлов даже при ошибке соединения
        if os.path.exists(os.path.join("vector_store", "index.faiss")):
            actual_size = get_kb_size()
            size_text = f"{actual_size:.2f} MB" if actual_size is not None else "N/A"
            return f"⚠️ Ошибка соединения, но база знаний существует\nРазмер: {size_text}"
        return f"❌ Ошибка проверки статуса: {str(e)}"

# Add this before creating the Gradio interface
def respond(message, chat_history, conv_id=None):
    """Handle chat responses"""
    if not message:
        return chat_history, conv_id
    
    # Add user message to chat history
    chat_history = chat_history + [[message, None]]
    
    try:
        # Send request to FastAPI backend
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={
                "message": message,
                "conversation_id": conv_id
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            bot_response = data["response"]
            conv_id = data["conversation_id"]
        else:
            bot_response = f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        bot_response = f"Error connecting to backend: {str(e)}"
    
    # Update last message with bot response
    chat_history[-1][1] = bot_response
    
    return chat_history, conv_id

def clear_history():
    """Clear chat history"""
    return None, None

# Initialize conversation ID
conversation_id = gr.State(None)

# Create the Gradio interface
with gr.Blocks(title="Status Law Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Status Law Assistant")
    
    with gr.Row():
        # Left column for chat (larger)
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=600,
                show_label=False,
                bubble_full_width=False
            )
            with gr.Row():
                msg = gr.Textbox(
                    label="Enter your question here",
                    placeholder="Type your message and press Enter...",
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Right column for controls (smaller)
        with gr.Column(scale=1):
            gr.Markdown("### Knowledge Base Controls")
            build_kb_btn = gr.Button("Create/Update Knowledge Base", variant="primary")
            check_status_btn = gr.Button("Check Status")
            kb_status = gr.Textbox(
                label="Knowledge Base Status",
                value="Checking status...",
                interactive=False
            )
            
            gr.Markdown("### Chat Controls")
            clear_btn = gr.Button("Clear Chat History")
            
            # Help section
            with gr.Accordion("ℹ️ How to use", open=False):
                gr.Markdown("""
                1. First, click **Create/Update Knowledge Base** button
                2. Wait for confirmation message
                3. Enter your question in the text field and press Enter or "Send" button
                4. Use "Clear Chat History" to start a new conversation
                """)
    
    # Event handlers remain the same
    msg.submit(respond, [msg, chatbot, conversation_id], [chatbot, conversation_id])
    submit_btn.click(respond, [msg, chatbot, conversation_id], [chatbot, conversation_id])
    clear_btn.click(clear_history, None, [chatbot, conversation_id])
    build_kb_btn.click(build_kb, inputs=None, outputs=kb_status)
    check_status_btn.click(check_kb_status, inputs=None, outputs=kb_status)

if __name__ == "__main__":
    # Launch Gradio interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
    #demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
