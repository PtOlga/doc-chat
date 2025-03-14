FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create directories with secure permissions
RUN mkdir -p cache/huggingface vector_store chat_history \
    && chown -R 1000:1000 . \
    && chmod -R 755 .

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY app.py .
COPY .env .
COPY index.html .

# Set environment variables
ENV HF_HOME=/app/cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/cache/huggingface
ENV XDG_CACHE_HOME=/app/cache
ENV PORT=8000

# Set permissions
RUN chown -R 1000:1000 /app \
    && find /app -type d -exec chmod 755 {} \; \
    && find /app -type f -exec chmod 644 {} \;

# Run as non-privileged user
USER 1000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Use a startup script
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]
