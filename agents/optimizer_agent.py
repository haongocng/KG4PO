import json
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.llm import LanguageModelManager

class OptimizerAgent:
    def __init__(self, provider="timelygpt"):
        """
        Initialize the Optimizer Agent for generating new prompt and lesson learned.
        """
        self.llm_manager = LanguageModelManager(provider=provider)
        self.llm = self.llm_manager.get_model(model_tier="json")

    def optimize(self, current_prompt: str, metrics: dict, failed_cases: list) -> dict:
        """
        Analyze current performance and generate an optimized system prompt
        :param current_prompt: The system prompt used in the current batch.
        :param metrics: Dictionary of evaluation scores.
        :param failed_cases: List of dictionaries containing details of failed cases.
        :return: A dictionary containing the new prompt, thought process, and lesson learned.
        """

        meta_system_template = """
        You are an expert Prompt Engineer specializing in LLM-based Recommender Systems.
        Your responsibility is to improve the System Prompt used by a recommendation model based on observed failures.
        
        CRITICAL CONTEXT:
        The target recommendation model is a RERANKER. It does NOT generate movies freely. Instead, it receives a user's session history and a strictly defined "Candidate Set" of 20 movies. Its job is to select and re-order the correct movies ONLY from that Candidate Set.
        
        INPUTS:
        You will receive:
        1. The current System Prompt.
        2. Evaluation metrics (NDCG, HIT, MAP).
        3. Several failed recommendation cases.

        OBJECTIVE:
        Improve the system prompt so that the recommender produces more accurate reranking of the candidate set.

        ANALYSIS STEPS:
        1. Analyze the failed cases and identify patterns of errors.
        2. Diagnose weaknesses in the current prompt (e.g., is it failing to capture long-term interests, or missing semantic links between the session and the target?).
        3. Propose targeted improvements to the prompt.
        4. Extract a general lesson learned from the failures.

        PROMPT EDITING STRATEGY:
        - Modify the existing prompt rather than rewriting it entirely when possible.
        - Preserve useful rules from the current prompt.
        - Focus on improving reasoning, intent extraction, and ranking criteria.

        CONSTRAINTS:
        - Do NOT include specific movie titles from failed cases inside the new prompt.
        - The new prompt MUST explicitly remind the model to strictly rerank items from the provided Candidate Set.
        - The new prompt must remain general and reusable.
        - Avoid overfitting to the specific examples.

        OUTPUT FORMAT (STRICT JSON):
        {{
        "thought_process": "Short explanation of the diagnosis and improvement strategy.",
        "new_system_prompt": "The complete improved system prompt.",
        "lesson_learned": "A short rule derived from the failure cases."
        }}
        """

        human_template = """
        --- BATCH PERFORMANCE REPORT ---
        
        1. CURRENT SYSTEM PROMPT:
        {current_prompt}
        
        2. ACHIEVED METRICS:
        {metrics_str}
        
        3. REPRESENTATIVE FAILED CASES (Ground truth was not in top predictions):
        {failed_cases_str}
        
        Perform your analysis and return the requested JSON.
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(meta_system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        chain = prompt | self.llm | StrOutputParser()

        metrics_str = json.dumps(metrics, indent=2)

        failed_cases_str = ""
        if not failed_cases:
            failed_cases_str = "No severe failures detected in this batch."
        else:
            # slice top 5 failed cases to prevent token limit and hallucination.
            for idx, fc in enumerate(failed_cases[:5]):
                session_hist = ", ".join(fc.get('session_raw', [])) if isinstance(fc.get('session_raw', []), list) else fc.get('session_raw', '')
                preds = ", ".join(fc.get('predictions', [])[:10]) if isinstance(fc.get('predictions', []), list) else fc.get('predictions', '')

                failed_cases_str += f" - Failure {idx+1}:\n"
                failed_cases_str += f"   + User history: {session_hist}\n"
                failed_cases_str += f"   + Ground truth (Actual next item): {fc.get('target_raw','')}\n"
                failed_cases_str += f"   + System's top predictions: {preds}\n\n"
        
        try: 
            raw_response = chain.invoke({
                "current_prompt": current_prompt,
                "metrics_str": metrics_str,
                "failed_cases_str":failed_cases_str
            })

            parsed_json = json.loads(raw_response)
            return {
                "new_prompt": parsed_json.get("new_system_prompt", current_prompt),
                "lesson_learned": parsed_json.get("lesson_learned", "Pay closer attention to user semantic preferences."),
                "thought": parsed_json.get("thought_process", "No reasoning provided.")
            }
        except json.JSONDecodeError:
            print("OPTIMIZER ERROR: The LLM did not return valid JSON format.")
            return {"new_prompt": current_prompt, "lesson_learned": "", "thought": "JSONDecodeError encountered."}
        except Exception as e:
            print(f"[Optimizer Error]: Unknown error during execution: {e}")
            return {"new_prompt": current_prompt, "lesson_learned": "", "thought": str(e)}

if __name__ == "__main__":
    optimizer = OptimizerAgent(provider="timelygpt") 
    
    dummy_prompt = "Suggest 20 movies based on the user's watch history."
    dummy_metrics = {"NDCG@10": 0.12, "HIT@10": 0.25, "MAP@10": 0.08}
    dummy_failures = [
        {
            "session_raw": ["The Body Snatcher", "Dracula"],
            "target_raw": "Frankenstein",
            "predictions": ["Toy Story", "Dumb & Dumber", "Aladdin"]
        }
    ]
    
    print("Running Optimizer Agent...")
    result = optimizer.optimize(dummy_prompt, dummy_metrics, dummy_failures)
    
    print("\n--- THOUGHT PROCESS ---")
    print(result["thought"])
    print("\n--- LESSON LEARNED ---")
    print(result["lesson_learned"])
    print("\n--- NEW SYSTEM PROMPT ---")
    print(result["new_prompt"])