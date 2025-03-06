#!/bin/bash

# Запуск Streamlit и FastAPI параллельно
streamlit run app.py &          # Запуск чат-бота
uvicorn api.main:app --reload   # Запуск API для анализа логов