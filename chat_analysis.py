import pandas as pd
from datetime import datetime
import json
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers import BaseTracer
from dataclasses import dataclass, asdict
import plotly.express as px
import streamlit as st

@dataclass
class ChatAnalysis:
    timestamp: str
    user_input: str
    bot_response: str
    context: str
    kb_version: str
    response_time: float
    tokens_used: int
    context_relevance_score: float

class ChatAnalyzer(BaseTracer):
    def __init__(self):
        super().__init__()
        self.analyses: List[ChatAnalysis] = []
        
    def load_logs(self, log_file_path: str) -> List[Dict]:
        """Загрузка и парсинг логов чата из JSON файла"""
        logs = []
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return logs

    def analyze_interaction(self, log_entry: Dict) -> ChatAnalysis:
        """Анализ одного взаимодействия в чате"""
        # Расчет базовых метрик
        timestamp = datetime.fromisoformat(log_entry["timestamp"])
        
        # Расчет времени ответа (можно заменить на реальную логику измерения)
        response_time = len(log_entry["bot_response"]) * 0.01  # Простая аппроксимация
        
        # Подсчет использованных токенов (заменить на реальный подсчет)
        tokens_used = len(log_entry["bot_response"].split()) + len(log_entry["user_input"].split())
        
        # Расчет релевантности контекста
        context_relevance = self._calculate_context_relevance(
            log_entry["user_input"],
            log_entry["context"],
            log_entry["bot_response"]
        )
        
        return ChatAnalysis(
            timestamp=timestamp.isoformat(),
            user_input=log_entry["user_input"],
            bot_response=log_entry["bot_response"],
            context=log_entry["context"],
            kb_version=log_entry["kb_version"],
            response_time=response_time,
            tokens_used=tokens_used,
            context_relevance_score=context_relevance
        )

    def _calculate_context_relevance(self, query: str, context: str, response: str) -> float:
        """Расчет оценки релевантности между запросом и предоставленным контекстом"""
        # Простая реализация - можно заменить на более сложную систему оценки
        query_terms = set(query.lower().split())
        context_terms = set(context.lower().split())
        response_terms = set(response.lower().split())
        
        query_context_overlap = len(query_terms & context_terms)
        context_response_overlap = len(context_terms & response_terms)
        
        if not query_terms or not context_terms:
            return 0.0
            
        return (query_context_overlap + context_response_overlap) / (len(query_terms) + len(context_terms))

    def create_analysis_dashboard(self):
        """Создание дашборда анализа чата в Streamlit"""
        st.title("Панель анализа чата")
        
        # Преобразование анализа в DataFrame
        df = pd.DataFrame([asdict(a) for a in self.analyses])
        
        # Базовая статистика
        st.header("Обзор")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Всего взаимодействий", len(df))
        with col2:
            st.metric("Среднее время ответа", f"{df['response_time'].mean():.2f}с")
        with col3:
            st.metric("Средняя релевантность контекста", f"{df['context_relevance_score'].mean():.2%}")
        with col4:
            st.metric("Всего использовано токенов", df['tokens_used'].sum())
            
        # Анализ временных рядов
        st.header("Тренды взаимодействий")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        fig = px.line(df, x='timestamp', y='response_time', title='Время ответа с течением времени')
        st.plotly_chart(fig)
        
        # Распределение релевантности контекста
        fig = px.histogram(df, x='context_relevance_score', 
                         title='Распределение оценок релевантности контекста',
                         nbins=20)
        st.plotly_chart(fig)
        
        # Детальные логи
        st.header("Детальные логи взаимодействий")
        st.dataframe(df)

def setup_chat_analysis():
    """Инициализация и настройка системы анализа чата"""
    analyzer = ChatAnalyzer()
    
    # Добавление к существующему логированию
    def enhanced_log_interaction(user_input: str, bot_response: str, context: str):
        # Ваш существующий код логирования
        log_interaction(user_input, bot_response, context)
        
        # Добавление анализа
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "context": context,
            "kb_version": st.session_state.kb_info['version']
        }
        analysis = analyzer.analyze_interaction(log_entry)
        analyzer.analyses.append(analysis)
    
    return analyzer, enhanced_log_interaction