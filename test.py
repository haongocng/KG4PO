import os
import json
import re
import pandas as pd
from tqdm import tqdm

from agents.recommender_agent import RecommenderAgent
from core.metrics import evaluate_batch, get_rank 
from core.error_retriever import ErrorMemoryBank

def load_jsonl_dataset(filepath):
    """Load the entire test dataset into memory."""
    dataset = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            dataset.append(json.loads(line))
    return dataset

def test(
    test_file="D:\\H_temp\\KG4PO\\data\\games\\test.context.jsonl",
    provider="timelygpt"
):
    print("=== STARTING INFERENCE PHASE ===")
    
    prompt_path = "prompts/best_system_prompt.txt"
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Cannot find the optimized prompt at {prompt_path}. Please run train.py first.")
        
    with open(prompt_path, 'r', encoding='utf-8') as f:
        best_prompt = f.read()
        
    print("[INFO] Successfully loaded Best System Prompt.")

    recommender = RecommenderAgent(provider=provider)
    error_bank = ErrorMemoryBank(reset=False)
    
    print(f"[INFO] Loading test dataset from {test_file}...")
    try:
        test_dataset = load_jsonl_dataset(test_file)
    except FileNotFoundError:
        print(f"[ERROR] Test file not found at {test_file}. Please check the path.")
        return

    predictions_list = []
    ground_truths_list = []
    
    for item in tqdm(test_dataset, desc="Evaluating Test Dataset"):
        input_str = item.get("input", "")
        ground_truth = item.get("target", "")
        knowledge = item.get("retrieved_knowledge", "")

        parts = input_str.split('\nCandidate Set:')
        session_str = parts[0].replace('Current session interactions:', '').strip()
        candidate_str = parts[1].strip() if len(parts) > 1 else ""

        session_items = re.findall(r'\d+\."([^"]+)"', session_str)
        
        past_errors_hints = error_bank.retrieve_similar_errors(session_items, top_k=2)
        
        preds = recommender.predict(
            system_prompt=best_prompt,
            session_items=session_str,      
            candidate_set=candidate_str,    
            retrieved_knowledge=knowledge,
            past_errors=past_errors_hints
        )
        
        predictions_list.append(preds)
        ground_truths_list.append(ground_truth)
        
    print("\n[INFO] Calculating Final Metrics on Test Set...")
    final_metrics = evaluate_batch(predictions_list, ground_truths_list, candidate_size=20)
    
    valid_count = sum(1 for p, g in zip(predictions_list, ground_truths_list) if get_rank(p, g) <= 20)
    
    print("\n=========================================================================")
    print("                           FINAL TEST RESULTS                            ")
    print("=========================================================================")
    print("KPI@K,1                 ,5                 ,10                ,20,                #valid_data")
    print(f"NDCG,{final_metrics.get('NDCG@1', 0)},{final_metrics.get('NDCG@5', 0)},{final_metrics.get('NDCG@10', 0)},{final_metrics.get('NDCG@20', 0)},{valid_count}")
    print(f"HIT,{final_metrics.get('HIT@1', 0)},{final_metrics.get('HIT@5', 0)},{final_metrics.get('HIT@10', 0)},{final_metrics.get('HIT@20', 0)},0")
    print(f"MAP,{final_metrics.get('MAP@1', 0)},{final_metrics.get('MAP@5', 0)},{final_metrics.get('MAP@10', 0)},{final_metrics.get('MAP@20', 0)},0")
    print("=========================================================================\n")

    metrics_summary_df = pd.DataFrame({
        "KPI@K": ["NDCG", "HIT", "MAP"],
        "1": [final_metrics.get('NDCG@1', 0), final_metrics.get('HIT@1', 0), final_metrics.get('MAP@1', 0)],
        "5": [final_metrics.get('NDCG@5', 0), final_metrics.get('HIT@5', 0), final_metrics.get('MAP@5', 0)],
        "10": [final_metrics.get('NDCG@10', 0), final_metrics.get('HIT@10', 0), final_metrics.get('MAP@10', 0)],
        "20": [final_metrics.get('NDCG@20', 0), final_metrics.get('HIT@20', 0), final_metrics.get('MAP@20', 0)],
        "#valid_data": [valid_count, 0, 0] 
    })

    metrics_csv_path = "results/games/gpt4.csv"
    os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
    metrics_summary_df.to_csv(metrics_csv_path, index=False)
    print(f"[INFO] Metrics summary saved successfully to {metrics_csv_path}")
    
    results_df = pd.DataFrame({
        "Input_String": [item.get("input", "") for item in test_dataset],
        "Ground_Truth": ground_truths_list,
        "Top_20_Predictions": predictions_list
    })
    
    results_path = "results/games/test_results_log.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"[INFO] Raw predictions saved to {results_path}")

if __name__ == "__main__":
    test(provider="timelygpt")