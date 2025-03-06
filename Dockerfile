FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create directories and set permissions
RUN mkdir -p /app/vector_store /app/chat_history /app/.cache && \
    chmod 777 /app/vector_store /app/chat_history /app/.cache

# Set environment variables
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache
ENV XDG_CACHE_HOME=/app/.cache

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set permissions for the application directory
RUN chown -R 1000:1000 /app && \
    chmod -R 755 /app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application as non-root user
USER 1000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]