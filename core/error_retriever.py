import os
import json
from rank_bm25 import BM25Okapi

class ErrorMemoryBank:
    def __init__(self, file_path="data/games/error_bank.json", reset=True):
        """
        Store and retrieve prediction errors with lessons learned using BM25
        """
        self.file_path = file_path
        self.errors = []
        self.bm25 = None
        if reset and os.path.exists(self.file_path):
            os.remove(self.file_path)

        self._load()
        self._build_bm25()

    def _tokenize(self, session_items: list) -> list:
        """
        Convert list items into list of tokens 
        Ex: ["The Body Snatcher", "Titanic"] -> ["the", "body", "snatcher", "titanic"]
        """
        text = " ".join(session_items).lower()
        return text.split()
    
    def _build_bm25(self):
        if not self.errors:
            self.bm25 = None
            return

        tokenized_corpus = [self._tokenize(err["session_items"]) for err in self.errors]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=1.2, b=0.5, epsilon=0.25)

    def add_error(self, session_items: list, ground_truth: str, predictions: list, lesson_learned: str):
        """
        Add a new error record to the bank & update the BM25 search engine.
        """
        record = {
            "session_items": session_items,
            "ground_truth": ground_truth,
            "predictions": predictions[:5],
            "lesson_learned": lesson_learned
        }

        self.errors.append(record)
        self._save()
        self._build_bm25()

    def retrieve_similar_errors(self, current_session: list, top_k=2) -> str:
        """
        Retrieve past lessons similar to the current session using BM25.
        """
        if not self.bm25 or not self.errors:
            return "There is no historical error information for this session."
        
        # Tokenize the current session to make a query
        tokenized_query = self._tokenize(current_session)
        
        # Calculate BM25 score for the error bank
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Sort the errors by descending BM25 score
        scored_errors = sorted(range(len(self.errors)), key=lambda i: doc_scores[i], reverse=True)
        
        # Filter to get the top K errors with scores > 0
        top_errors = [self.errors[idx] for idx in scored_errors[:top_k] if doc_scores[idx] > 0]

        # Fallback: if no similar errors found, return the latest errors
        if not top_errors:
            top_errors = self.errors[-top_k:]

        # Format the retrieved errors info
        result_str = ""
        for idx, err in enumerate(top_errors):
            result_str += f"[Similar error {idx+1}]\n"
            result_str += f"- History interactions: {' -> '.join(err['session_items'])}\n"
            result_str += f"- Ground truth: {err['ground_truth']}\n"
            result_str += f"- Wrong prediction: {', '.join(err['predictions'])}\n"
            result_str += f"- Lesson learned: {err['lesson_learned']}\n\n"

        return result_str.strip()
    
    def _save(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.errors, f, ensure_ascii=False, indent=4)

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.errors = json.load(f)
            except Exception as e:
                self.errors = []

