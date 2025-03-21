import os
import sys
import threading
import time
import gradio as gr
import uvicorn
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Add project root to Python path
current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our main application
from api.fastapi_server import app as fastapi_app

# Run FastAPI server in a separate thread
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

# Start FastAPI in a background thread
fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
fastapi_thread.start()

# Wait for FastAPI to start
time.sleep(5)

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

# Добавим функцию проверки статуса базы знаний
def check_kb_status():
    try:
        response = requests.get("http://127.0.0.1:8000/")
        if response.status_code == 200:
            data = response.json()
            if data["knowledge_base_exists"]:
                kb_info = data["kb_info"]
                # Добавляем проверки на None и значения по умолчанию
                version = kb_info.get('version', 'N/A')
                size = kb_info.get('size', 0)
                return f"✅ База знаний готова к работе\nВерсия: {version}\nРазмер: {size:.2f if size else 0} MB"
            else:
                return "❌ База знаний не создана. Нажмите кнопку 'Create/Update Knowledge Base'"
    except Exception as e:
        return f"❌ Ошибка проверки статуса: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Status Law Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Status Law Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Кнопки управления базой знаний
            build_kb_btn = gr.Button("Create/Update Knowledge Base", variant="primary")
            check_status_btn = gr.Button("Check Status")
            kb_status = gr.Textbox(
                label="Knowledge Base Status",
                value="Checking status...",
                interactive=False
            )
            # Привязываем обе кнопки
            build_kb_btn.click(build_kb, inputs=None, outputs=kb_status)
            check_status_btn.click(check_kb_status, inputs=None, outputs=kb_status)
    
    gr.Markdown("### 💬 Chat Interface")
    conversation_id = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Улучшенный интерфейс чата
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                show_label=False,
                bubble_full_width=False
            )
            with gr.Row():
                msg = gr.Textbox(
                    label="Введите ваш вопрос здесь",
                    placeholder="Напишите ваш вопрос и нажмите Enter...",
                    scale=4
                )
                submit_btn = gr.Button("Отправить", variant="primary", scale=1)
            
            # Добавляем очистку истории
            clear_btn = gr.Button("Очистить историю")
            
            def clear_history():
                return [], None
            
            def respond(message, chat_history, conv_id):
                if not message.strip():
                    return chat_history, conv_id
                
                chat_history.append([message, ""])
                response, new_conv_id = chat_with_api(message, conv_id)
                chat_history[-1][1] = response
                return chat_history, new_conv_id
            
            # Привязываем обработчики
            msg.submit(respond, [msg, chatbot, conversation_id], [chatbot, conversation_id])
            submit_btn.click(respond, [msg, chatbot, conversation_id], [chatbot, conversation_id])
            clear_btn.click(clear_history, None, [chatbot, conversation_id])

    # Добавляем информацию об использовании
    with gr.Accordion("ℹ️ Как использовать", open=False):
        gr.Markdown("""
        1. Сначала нажмите кнопку **Create/Update Knowledge Base** для создания базы знаний
        2. Дождитесь сообщения об успешном создании базы
        3. Введите ваш вопрос в текстовое поле и нажмите Enter или кнопку "Отправить"
        4. Используйте кнопку "Очистить историю" для начала новой беседы
        """)

if __name__ == "__main__":
    # Проверяем статус базы знаний при запуске
    initial_status = check_kb_status()
    # Launch Gradio interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
    #demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
