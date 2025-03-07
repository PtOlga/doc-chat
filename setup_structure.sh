# Создание основной структуры директорий
mkdir -p api utils tests chat_history vector_store

# Создание файлов Python
touch api/__init__.py
touch utils/__init__.py
touch tests/__init__.py
touch tests/test_app.py

# Перемещение существующих файлов
mv chat_analysis.py utils/
mv analysis.py api/

# Создание .gitkeep для пустых директорий
touch chat_history/.gitkeep
touch vector_store/.gitkeep