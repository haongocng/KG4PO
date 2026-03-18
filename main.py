import json
import os
from tqdm import tqdm
from agents.recommender_agent import RecommenderAgent
from agents.optimizer_agent import OptimizerAgent
from core.metrics import evaluate_batch
from core.memory import TrajectoryBuffer
from core.error_retriever import ErrorMemoryBank

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
    train_file = ".//data//ml_100k//context//train.context.jsonl",
    batch_size = 10,
    max_batches = 50,
    provider= "timelygpt"
):
    print("STARTING PROMPT OPTIMIZATION TRAINING ...")

    # 1. Initialize core module
    trajectory_buffer = TrajectoryBuffer()
    error_bank = ErrorMemoryBank()

    # 2. Initialize agents
    recommender = RecommenderAgent(provider=provider)
    optimizer = OptimizerAgent(provider=provider)
    
    # 3. Initial system prompt
    current_prompt = """
    Based on the user's current session interactions and any available additional context, such as browsing history, search queries, or demographic information, you need to answer the following subtasks step by step:\n\
    1. Discover combinations of items within the session, where the number of combinations can be one or more, and consider how these combinations might relate to the user's broader interests and preferences.\n\
    2. For each combination, analyze the items and their attributes (e.g., genre, director, release year) to infer the user's interactive intent within each combination. Consider how the user's interactions with these items might reflect their underlying preferences or interests.\n\
    3. Select the intent from the inferred ones that best represents the user's current preferences, taking into account the user's behavior and any available additional context. If multiple intents seem plausible, consider the strength of the evidence for each intent and choose the one that is most strongly supported.\n\
    4. Based on the selected intent, rerank the 20 items in the candidate set according to their relevance and potential appeal to the user, considering factors such as genre, tone, and similarity to the items in the current session. Provide a clear ranking of all 20 items, with the most relevant items first.\n\
    Note that the order of all items in the candidate set must be given, and the items for ranking must be within the candidate set. When reranking, consider the trade-off between exploiting the user's current preferences and exploring new items that might be of interest.\n
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
            session_items = item.get("session", [])
            ground_truth = item.get("ground_truth", "")
            knowledge = item.get("retrieved_knowledge", "")

            # Retrive similar pass errors from Error Bank.
            past_errors = error_bank.retrieve_similar_errors(session_items, top_k=2)

            preds = recommender.predict(
                system_prompt=current_prompt,
                session_items=session_items,
                retrieved_knowledge=knowledge,
                past_errors=past_errors
            )

            predictions_list.append(preds)
            ground_truths_list.append(ground_truth)

            preds_lower = [p.lower() for p in preds]
            if ground_truth.lower() not in preds_lower:
                failed_cases.append({
                    "session_raw": session_items,
                    "target_raw": ground_truth,
                    "predictions": preds[:10]
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
    
    best_record = trajectory_buffer.get_best_record(target_metrics="NDCG@10")
    if best_record:
        best_prompt_path = "prompts/best_system_prompt.txt"
        os.makedirs(os.path.dirname(best_prompt_path), exist_ok=True)
        with open(best_prompt_path, 'w') as f:
            f.write(best_record['prompt'])
        print(f"\nTraining Complete. Best Prompt saved to {best_prompt_path}")
        print(f"Best Metrics: {best_record['metrics']}")

if __name__ == "__main__":
    train(batch_size=5, max_batches=10, provider="timely")