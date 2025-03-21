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

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Status Law Assistant")
    
    with gr.Row():
        with gr.Column():
            build_kb_btn = gr.Button("Create/Update Knowledge Base")
            kb_status = gr.Textbox(label="Knowledge Base Status")
            build_kb_btn.click(build_kb, inputs=None, outputs=kb_status)
    
    conversation_id = gr.State(None)
    
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label="Chat with Assistant")
            msg = gr.Textbox(label="Your Question")
            
            def respond(message, chat_history, conv_id):
                if not message.strip():
                    return chat_history, conv_id
                
                chat_history.append([message, ""])
                response, new_conv_id = chat_with_api(message, conv_id)
                chat_history[-1][1] = response
                return chat_history, new_conv_id
            
            msg.submit(respond, [msg, chatbot, conversation_id], [chatbot, conversation_id])

if __name__ == "__main__":
    # Launch Gradio interface
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
