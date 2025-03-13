FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create directories and set permissions
RUN mkdir -p /app/vector_store /app/chat_history /app/.cache /app/logs && \
    chmod 777 /app/vector_store /app/chat_history /app/.cache /app/logs

# Set environment variables
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV XDG_CACHE_HOME=/app/.cache
ENV PYTHONUNBUFFERED=1

# Create cache directories with proper permissions
RUN mkdir -p /app/.cache/huggingface && \
    chmod -R 777 /app/.cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R 1000:1000 /app && \
    chmod -R 755 /app

EXPOSE 8000

USER 1000

# Изменяем команду запуска для сохранения логов
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 --log-level debug 2>&1 | tee /app/logs/app.log"]
