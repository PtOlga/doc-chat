import pytest
from fastapi.testclient import TestClient
from app import app
import os
import json
from utils import ChatAnalyzer
from api import LogAnalyzer

client = TestClient(app)

# Фикстуры для тестов
@pytest.fixture
def test_log_file(tmp_path):
    """Создает временный файл с тестовыми логами"""
    log_file = tmp_path / "test_chat_logs.json"
    test_logs = [
        {
            "timestamp": "2024-01-01T12:00:00",
            "user_input": "Test question",
            "bot_response": "Test response",
            "context": "Test context",
            "kb_version": "20240101"
        }
    ]
    
    with open(log_file, 'w', encoding='utf-8') as f:
        for log in test_logs:
            f.write(json.dumps(log) + '\n')
    return log_file

# Тесты API endpoints
def test_chat_endpoint():
    """Тест основного эндпоинта чата"""
    response = client.post(
        "/chat",
        json={"message": "What services do you provide?"}
    )
    assert response.status_code == 200
    assert "response" in response.json()

def test_analysis_basic():
    """Тест эндпоинта базового анализа"""
    response = client.get("/api/analysis/basic")
    assert response.status_code == 200
    data = response.json()
    assert "total_interactions" in data

def test_analysis_temporal():
    """Тест эндпоинта временного анализа"""
    response = client.get("/api/analysis/temporal")
    assert response.status_code == 200
    data = response.json()
    assert "daily_activity" in data
    assert "hourly_pattern" in data

# Тесты компонентов
def test_chat_analyzer():
    """Тест класса ChatAnalyzer"""
    analyzer = ChatAnalyzer()
    assert hasattr(analyzer, 'analyze_interaction')
    assert hasattr(analyzer, 'create_analysis_dashboard')

def test_log_analyzer(test_log_file):
    """Тест класса LogAnalyzer"""
    analyzer = LogAnalyzer(log_path=str(test_log_file))
    stats = analyzer.get_basic_stats()
    assert "total_interactions" in stats
    assert stats["total_interactions"] == 1

# Тесты утилит
def test_knowledge_base():
    """Тест работы с базой знаний"""
    vector_store_path = "vector_store"
    assert os.path.exists(vector_store_path)

def test_environment():
    """Тест настройки окружения"""
    assert "GROQ_API_KEY" in os.environ

# Тесты обработки ошибок
def test_chat_endpoint_error():
    """Тест обработки ошибок в чате"""
    response = client.post(
        "/chat",
        json={"message": ""}  # Пустой запрос
    )
    assert response.status_code == 422  # Validation error

def test_analysis_error():
    """Тест обработки ошибок в анализе"""
    # Временно меняем путь к логам на несуществующий
    original_path = LogAnalyzer._default_log_path
    LogAnalyzer._default_log_path = "nonexistent/path.json"
    
    response = client.get("/api/analysis/basic")
    assert response.status_code == 200
    assert response.json() == {}
    
    # Восстанавливаем оригинальный путь
    LogAnalyzer._default_log_path = original_path

if __name__ == "__main__":
    pytest.main([__file__])