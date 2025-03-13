FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создание директорий с безопасными правами
RUN mkdir -p cache/huggingface vector_store chat_history \
    && chown -R 1000:1000 . \
    && chmod -R 755 .

# Копируем зависимости отдельно для кэширования
COPY requirements.txt .

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Настройка переменных окружения
ENV TRANSFORMERS_CACHE=/app/cache/huggingface
ENV HF_HOME=/app/cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/cache/huggingface
ENV XDG_CACHE_HOME=/app/cache

# Фиксируем права (только для вновь созданных файлов)
RUN chown -R 1000:1000 /app \
    && find /app -type d -exec chmod 755 {} \; \
    && find /app -type f -exec chmod 644 {} \;

# Запускаем от непривилегированного пользователя
USER 1000

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]