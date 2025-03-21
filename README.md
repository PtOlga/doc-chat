---
title: 'Doc LS Chatbot '
emoji: 🔥
colorFrom: yellow
colorTo: yellow
sdk: docker
sdk_version: 1.42.2
app_file: app.py
pinned: false
short_description: It is a chat built with an AI model about www.Status.law
---

# LS DOC Chatbot Log

It is a chat app built using Hugging Face and Docker Space that allows users to interact with an AI model to communicate about www.Status.law

This application provides two interfaces:
1. Web Interface (accessible via /web endpoint)
2. Hugging Face Spaces Interface (using Gradio)

## Access Points
- Web Interface: http://localhost:8000/web
- Gradio Interface: http://localhost:7860
- API Endpoints: http://localhost:8000/docs

## Environment Variables
Required environment variables:
- GROQ_API_KEY
- HF_TOKEN (optional, for Hugging Face integration)