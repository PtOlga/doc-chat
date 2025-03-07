import pandas as pd
from datetime import datetime
import json
from typing import List, Dict
from dataclasses import dataclass, asdict

@dataclass
class ChatAnalysis:
    """Class for storing analysis results of a single chat interaction"""
    timestamp: str
    user_input: str
    bot_response: str
    context: str
    kb_version: str
    response_time: float
    tokens_used: int
    context_relevance_score: float

class ChatAnalyzer:
    """Class for analyzing chat interactions"""
    def __init__(self):
        self.analyses: List[ChatAnalysis] = []
        
    def load_logs(self, log_file_path: str) -> List[Dict]:
        """Load and parse chat logs from JSON file"""
        logs = []
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return logs

    def analyze_interaction(self, log_entry: Dict) -> ChatAnalysis:
        """Analyze a single chat interaction"""
        timestamp = datetime.fromisoformat(log_entry["timestamp"])
        response_time = len(log_entry["bot_response"]) * 0.01
        tokens_used = len(log_entry["bot_response"].split()) + len(log_entry["user_input"].split())
        
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
        """Calculate relevance score between query and provided context"""
        query_terms = set(query.lower().split())
        context_terms = set(context.lower().split())
        response_terms = set(response.lower().split())
        
        query_context_overlap = len(query_terms & context_terms)
        context_response_overlap = len(context_terms & response_terms)
        
        if not query_terms or not context_terms:
            return 0.0
            
        return (query_context_overlap + context_response_overlap) / (len(query_terms) + len(context_terms))

    def get_analysis_data(self) -> Dict:
        """Get aggregated analysis data"""
        df = pd.DataFrame([asdict(a) for a in self.analyses])
        
        if df.empty:
            return {
                "total_interactions": 0,
                "avg_response_time": 0,
                "avg_relevance": 0,
                "total_tokens": 0
            }
        
        return {
            "total_interactions": len(df),
            "avg_response_time": float(df['response_time'].mean()),
            "avg_relevance": float(df['context_relevance_score'].mean()),
            "total_tokens": int(df['tokens_used'].sum())
        }

def setup_chat_analysis():
    """Initialize and configure chat analysis system"""
    analyzer = ChatAnalyzer()
    
    def enhanced_log_interaction(user_input: str, bot_response: str, context: str):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "context": context,
            "kb_version": datetime.now().strftime("%Y%m%d-%H%M%S")
        }
        
        analysis = analyzer.analyze_interaction(log_entry)
        analyzer.analyses.append(analysis)
        
        with open("chat_history/chat_logs.json", "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")
    
    return analyzer, enhanced_log_interaction