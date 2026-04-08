import json
import os
import re
from tqdm import tqdm
from agents.recommender_agent import RecommenderAgent
from agents.optimizer_agent import OptimizerAgent
from core.metrics import evaluate_batch
from core.memory import TrajectoryBuffer
from core.error_retriever import ErrorMemoryBank
from core.metrics import evaluate_batch, get_rank

def load_jsonl_batches(filepath, batch_size):
    batch = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch    

def train(
    train_file = ".//data//games//train.context.jsonl",
    batch_size = 10,
    max_batches = 50,
    provider= "timelygpt"
):
    print("STARTING PROMPT OPTIMIZATION TRAINING ...")

    # 1. Initialize core module
    trajectory_buffer = TrajectoryBuffer(reset=True) 
    error_bank = ErrorMemoryBank(reset=True)

    # 2. Initialize agents
    recommender = RecommenderAgent(provider=provider)
    optimizer = OptimizerAgent(provider=provider)
    
    # 3. Initial system prompt
    current_prompt = """
    Based on the user's current session interactions, you need to answer the following subtasks step by step:
    1. Discover combinations of items within the session, where the number of combinations can be one or more.
    2. Based on the items within each combination, infer the user's interactive intent within each combination.
    3. Select the intent from the inferred ones that best represents the user's current preferences.
    4. Based on the selected intent, please rerank the 20 items in the candidate set according to the possibility of potential user interactions and show me your ranking results with item index.
    Note that the order of all items in the candidate set must be given, and the items for ranking must be within the candidate set.
    """

    latest_record = trajectory_buffer.get_latest_record()
    if latest_record:
        print("Resuming from previous prompt in Trajectory Buffer.")
        current_prompt = latest_record['prompt']

    batch_generator = load_jsonl_batches(train_file, batch_size)

    for batch_idx, batch in enumerate(batch_generator):
        if max_batches is not None and batch_idx >= max_batches:
            print(f"Reached maximum batches limit ({max_batches}). Stopping training.")
            break

        print(f"\n ----processing Batch {batch_idx + 1}----")

        predictions_list = []
        ground_truths_list = []
        failed_cases = []

        for item in tqdm(batch, desc="Recommender predicting"):
            input_str = item.get("input", "")
            ground_truth = item.get("target", "")
            knowledge = item.get("retrieved_knowledge", "")

            parts = input_str.split('\nCandidate Set:')
            session_str = parts[0].replace('Current session interactions:', '').strip()
            candidate_str = parts[1].strip() if len(parts) > 1 else ""

            # Trích xuất dạng list để phục vụ Error Bank
            session_items = re.findall(r'\d+\."([^"]+)"', session_str)

            # Retrive similar pass errors from Error Bank.
            past_errors = error_bank.retrieve_similar_errors(session_items, top_k=2)

            preds = recommender.predict(
                system_prompt=current_prompt,
                session_items=session_str,     
                candidate_set=candidate_str,
                retrieved_knowledge=knowledge,
                past_errors=past_errors
            )

            predictions_list.append(preds)
            ground_truths_list.append(ground_truth)

            ACCEPTABLE_RANK = 8

            rank = get_rank(preds, ground_truth)

            if rank > ACCEPTABLE_RANK:
                failed_cases.append({
                    "session_raw": session_items,
                    "target_raw": ground_truth,
                    "predictions": preds[:10],
                    "actual_rank": rank
                })

        # Evaluate batch performance
        metrics_dict = evaluate_batch(predictions_list, ground_truths_list, candidate_size=20)
        print(f"Metrics for batch {batch_idx + 1}: {metrics_dict}")

        # OPTIMIZE PROMPT AND EXTRACT LESSONS
        if failed_cases:
            print("Optimizing prompt and extracting lessons from failed cases ...")
            optimization_result = optimizer.optimize(
                current_prompt=current_prompt,
                metrics=metrics_dict,
                failed_cases=failed_cases
            )

            current_prompt = optimization_result['new_prompt']
            print(f"Thought Process: {optimization_result['thought']}")
            print(f"Lesson Learned: {optimization_result['lesson_learned']}")

            rep_case = failed_cases[0]
            error_bank.add_error(
                session_items=rep_case['session_raw'],
                ground_truth=rep_case['target_raw'],
                predictions=rep_case['predictions'],
                lesson_learned=optimization_result['lesson_learned']
            )
        else:
            print("No failures detected in this batch. No optimization needed.")

        trajectory_buffer.add_record(
            prompt=current_prompt,
            metrics=metrics_dict,
            error_logs=failed_cases
        )
    
    best_record = trajectory_buffer.get_best_record(target_metric="NDCG@10")
    if best_record:
        best_prompt_path = "prompts/best_system_prompt.txt"
        os.makedirs(os.path.dirname(best_prompt_path), exist_ok=True)
        with open(best_prompt_path, 'w') as f:
            f.write(best_record['prompt'])
        print(f"\nTraining Complete. Best Prompt saved to {best_prompt_path}")
        print(f"Best Metrics: {best_record['metrics']}")

if __name__ == "__main__":
    train(batch_size=5, max_batches=None, provider="timelygpt")