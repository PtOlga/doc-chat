# analysis.py
import json
import pandas as pd
from collections import defaultdict
from typing import List, Dict
from datetime import datetime

class LogAnalyzer:
    def __init__(self, log_path: str = "chat_history/chat_logs.json"):
        self.log_path = log_path
        self.logs = self._load_logs()

    def _load_logs(self) -> List[Dict]:
        """Load and parse log entries from JSON file"""
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def get_basic_stats(self) -> Dict:
        """Calculate basic conversation statistics"""
        if not self.logs:
            return {}
            
        return {
            "total_interactions": len(self.logs),
            "unique_users": len({log.get('session_id') for log in self.logs}),
            "avg_response_length": pd.Series([len(log['bot_response']) for log in self.logs]).mean(),
            "most_common_questions": self._get_common_questions(),
            "knowledge_base_usage": self._calculate_kb_usage()
        }

    def _get_common_questions(self, top_n: int = 5) -> List[Dict]:
        """Identify most frequent user questions"""
        question_counts = defaultdict(int)
        for log in self.logs:
            question_counts[log['user_input']] += 1
        return sorted(
            [{"question": k, "count": v} for k, v in question_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:top_n]

    def _calculate_kb_usage(self) -> Dict:
        """Analyze knowledge base effectiveness"""
        context_usage = defaultdict(int)
        for log in self.logs:
            if log.get('context'):
                context_usage['with_context'] += 1
            else:
                context_usage['without_context'] += 1
        return context_usage

    def temporal_analysis(self) -> Dict:
        """Analyze usage patterns over time"""
        df = pd.DataFrame(self.logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return {
            "daily_activity": df.resample('D', on='timestamp').size().to_dict(),
            "hourly_pattern": df.groupby(df['timestamp'].dt.hour).size().to_dict()
        }

    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        stats = self.get_basic_stats()
        temporal = self.temporal_analysis()
        
        report = (
            "Legal Assistant Usage Report\n"
            "----------------------------\n"
            f"Period: {self.logs[0]['timestamp']} - {self.logs[-1]['timestamp']}\n\n"
            f"Total Interactions: {stats['total_interactions']}\n"
            f"Unique Users: {stats['unique_users']}\n"
            f"Average Response Length: {stats['avg_response_length']:.1f} chars\n\n"
            "Top Questions:\n"
            + "".join(f"- {q['question']}: {q['count']}\n" for q in stats['most_common_questions'])
            + "\nKnowledge Base Usage:\n"
            f"- With context: {stats['knowledge_base_usage'].get('with_context', 0)}\n"
            f"- Without context: {stats['knowledge_base_usage'].get('without_context', 0)}\n\n"
            "Usage Patterns:\n"
            f"- Daily Activity: {temporal['daily_activity']}\n"
            f"- Hourly Distribution: {temporal['hourly_pattern']}\n"
        )
        return report
