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
        You are a Prompt Engineer for LLM recommender systems.
        Your task is to improve the System Prompt for a recommendation model using observed failures.

        Context:
        The target recommendation model is a reranker. It does not generate games independently. It receives session history and a candidate set of 20 items. It selects and re-orders items from the candidate set.

        Inputs:
        1. Current System Prompt.
        2. Evaluation metrics.
        3. Failed recommendation cases.

        Objective:
        Improve the system prompt to increase the ranking quality of the candidate set.

        Analysis steps:
        1. Analyze failed cases and identify error patterns.
        2. Diagnose weaknesses in the current prompt.
        3. Propose improvements to the prompt.
        4. Extract a lesson learned from the failures.

        Editing strategy:
        - Modify the existing prompt instead of rewriting it.
        - Keep useful rules from the current prompt.
        - Focus on reasoning, intent extraction, and ranking criteria.

        CONSTRAINTS:
        - Do NOT include specific game titles from failed cases inside the new prompt.
        - The new prompt MUST explicitly remind the model to strictly rerank items from the provided Candidate Set.
        - The new prompt must remain general and reusable.
        - Avoid overfitting to the specific examples.

        # REFINED STRATEGY FOR LESSON LEARNED:
        A "lesson learned" is NOT general advice. It must be a SPECIFIC, REUSABLE HEURISTIC for ranking decisions that helps the model avoid repeating a mistake.

        A good lesson must follow this structure:
        [Pattern Detected] -> [Failure Cause] -> [Trigger Signal] -> [Actionable Ranking Rule].

        # STRICT REQUIREMENTS:
        - The lesson MUST be concise, atomic, and structured.
        - Avoid storytelling or long narrative examples.
        - The lesson MUST define a concrete ranking action (e.g., prioritize, downrank, filter, boost).
        - DO NOT produce vague lessons such as:
        "improve understanding", "consider preferences", "balance diversity".
        - The lesson MUST explicitly use Knowledge Graph attributes as ranking signals.
        - Include a "signal" field describing when this rule should be applied.

        Output format (JSON):
        {{
            "thought_process": "Explanation of the diagnosis and improvement strategy.",
            "new_system_prompt": "The complete improved system prompt.",
                "lesson_learned": {
                "pattern": "Short name of failure pattern",
                "signal": "Observable condition in session history",
                "failure_cause": "Why the model made the wrong ranking decision",
                "rule": "Specific ranking action using candidate set and KG signals",
                "applicability": "When this rule should be reused",
                "priority": "low | medium | high"
            }
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
            for idx, fc in enumerate(failed_cases[:6]):
                session_hist = ", ".join(fc.get('session_raw', [])) if isinstance(fc.get('session_raw', []), list) else fc.get('session_raw', '')
                preds = ", ".join(fc.get('predictions', [])[:20]) if isinstance(fc.get('predictions', []), list) else fc.get('predictions', '')

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