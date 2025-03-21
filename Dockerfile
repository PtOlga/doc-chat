FROM python:3.9-slim

WORKDIR /app

# Install system dependencies with verbose output
RUN set -x && \
    apt-get update && \
    apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && echo "System dependencies installed successfully"

# Create directories with secure permissions
RUN set -x && \
    mkdir -p cache/huggingface vector_store chat_history && \
    chown -R 1000:1000 . && \
    chmod -R 755 . && \
    echo "Directories created successfully"

# Copy requirements first for better caching
COPY requirements.txt .
RUN set -x && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "Python dependencies installed successfully"

# Copy application files
COPY app.py .
COPY index.html .
COPY api/ ./api/

# Set environment variables
ENV HF_HOME=/app/cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/cache/huggingface
ENV XDG_CACHE_HOME=/app/cache
ENV PORT=8000

# Set permissions
RUN set -x && \
    chown -R 1000:1000 /app && \
    find /app -type d -exec chmod 755 {} \; && \
    find /app -type f -exec chmod 644 {} \; && \
    echo "Permissions set successfully"

# Run as non-privileged user
USER 1000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Use a startup script with debug output
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
CMD ["python", "app.py"]

