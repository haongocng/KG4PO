import os
import json

class TrajectoryBuffer:
    def __init__(self, file_path="data/games/trajectory_history.json", reset=True):
        """
        Manage history of prompt optimization loops.
        :param file_path: Path to json file storing trajectory history.
        """
        self.file_path = file_path
        self.history = []
        if reset and os.path.exists(self.file_path):
            os.remove(self.file_path)

        self._load()

    def add_record(self, prompt: str, metrics: dict, error_logs: list):
        """
        Add a new record after each batch evaluation loop.
        :param prompt: the prompt used in the batch.
        :param metrics: evaluation metrics for the batch.
        :param error_logs: list of errors identified in the batch.
        """
        record = {
            "step": len(self.history) + 1,
            "prompt": prompt,
            "metrics": metrics,
            "errors": error_logs
        }
        self.history.append(record)
        self._save()

    def get_best_record(self, target_metric="NDCG@10"):
        """
        Retrieve the best record with the highest metric score.
        """
        if not self.history:
            return None
        best_record = max(self.history, key=lambda x: x["metrics"].get(target_metric, 0))
        return best_record
    
    def get_latest_record(self):
        """
        Retrieve the latest record.
        """
        if not self.history: return None
        return self.history[-1]
    
    def _save(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception as e:
                print(f"Error loading: {e}")
                self.history = []