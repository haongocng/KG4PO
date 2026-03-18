import json
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.llm import LanguageModelManager

class RecommenderAgent:
    def __init__(self, provider="openai"):
        """
        Khởi tạo Agent với mô hình ngôn ngữ chuyên biệt cho việc trả về JSON.
        """
        self.llm_manager = LanguageModelManager(provider=provider)
        self.llm = self.llm_manager.get_model(model_tier="json")
        
    def predict(self, system_prompt: str, session_items: list, retrieved_knowledge: str, past_errors: str = "") -> list:
        """
        predict next 20 items.
        
        :param system_prompt: Prompt is guided behavior by optimizer generated
        :param session_items: List items which user watched in current session (Short-term). 
        :param retrieved_knowledge: Knowledge retrieved from Knowledge Graph (Long-term).
        :param past_errors: từ Similar errors in the past from Error Bank.
        :return: List 20 items recommended.
        """
        
        human_template = """
        This is a movie recommendation task. You are an expert movie recommender system. Your task is to analyze the user's current session history, the retrieved knowledge from the Knowledge Graph, and any past errors to predict the next 20 movies that the user is most likely to watch.
        Here are the details you should consider:
        
        1. User interactions (Session):
        {session_items}
        
        2. Knowledge from Knowledge Graph (Context):
        {retrieved_knowledge}
        
        3. Experiences / Past mistakes should be avoided (Past Errors):
        {past_errors}
        Based on the instructions from System Prompt and the information above, suggest the next 20 correct movies.
        REQUIRED: Return JSON format with unique structure (without any other text):
        {{"recommendations": ["name movie", "name movie 2", ..., "name movie 20"]}}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("{system_prompt}"),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            session_str = " -> ".join(session_items)
            errors_str = past_errors if past_errors.strip() else "There is no historical error information for this session."
            
            raw_response = chain.invoke({
                "system_prompt": system_prompt,
                "session_items": session_str,
                "retrieved_knowledge": retrieved_knowledge,
                "past_errors": errors_str
            })
            
            parsed_json = json.loads(raw_response)
            recommendations = parsed_json.get("recommendations", [])
            
            return recommendations[:20]
            
        except json.JSONDecodeError:
            print("Lỗi: LLM không trả về định dạng JSON hợp lệ.")
            return []
        except Exception as e:
            print(f"Lỗi không xác định khi chạy RecommenderAgent: {e}")
            return []

if __name__ == "__main__":
    agent = RecommenderAgent(provider="timelygpt")
    
    # Dữ liệu giả lập từ file test.context.jsonl
    dummy_system_prompt = "You are an excellent movie recommendation system. Analyze history, knowledge and past errors to make the most accurate predictions."
    dummy_session = ["The Body Snatcher"]
    dummy_knowledge = "- Item 'The Body Snatcher' belong to category: Gothic Romances. Keywords: Horror, Thriller."
    dummy_errors = ""
    
    print("Đang chạy dự đoán...")
    results = agent.predict(dummy_system_prompt, dummy_session, dummy_knowledge, dummy_errors)
    print(f"Danh sách đề xuất ({len(results)} phim):")
    print(results)