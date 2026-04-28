import json
from datetime import datetime

class ActiveLearningLoop:
    def __init__(self, log_path='events.json'):
        self.log_path = log_path
        self.events = []

    def log_event(self, gtid: str, embedding: list, score: float, result_label: str):
        event = {
            "gtid": gtid,
            "score": score,
            "timestamp": datetime.utcnow().isoformat(),
            "device": "AOA-Core-Node",
            "result": result_label,
            "action": self._determine_action(score)
        }
        self.events.append(event)
        
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(event) + '\n')
            
    def _determine_action(self, score):
        if score >= 90: return 'auto-label'
        elif score >= 70: return 'human-review'
        else: return 'anomaly-flag'
