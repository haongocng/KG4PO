from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config import DEEPINFRA_API_KEY, DEEPINFRA_API_BASE_URL, OPENAI_API_KEY, TIMELYGPT_API_BASE_URL, TIMELYGPT_API_KEY
# from config import OPENAI_API_KEY, OPENAI_API_BASE_URL

class LanguageModelManager:
    def __init__(self, provider="openai"):
        self.llm = None
        self.power_llm = None
        self.json_llm = None
        self._initialize_llms(provider)

    def _initialize_llms(self, provider):
        try:
            if provider == "openai":
                self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0)
                
                self.power_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0.5)

                self.json_llm = ChatOpenAI(
                    api_key=OPENAI_API_KEY,
                    model="gpt-4o",
                    temperature=0.5,
                    model_kwargs={"response_format": {"type": "json_object"}}
                )
                print("LLM initialized successfully with OpenAI")
            
            elif provider == "deepinfra":
                common_params = {
                    "api_key": DEEPINFRA_API_KEY,
                    "base_url": DEEPINFRA_API_BASE_URL,
                    "model": "Qwen/Qwen3-Next-80B-A3B-Instruct"
                }
                self.llm = ChatOpenAI(**common_params, temperature=0)
                self.power_llm = ChatOpenAI(**common_params, temperature=0.5)
                self.json_llm = ChatOpenAI(**common_params, temperature=0.5,
                    model_kwargs={"response_format": {"type": "json_object"}}
                )
                print("LLM initialized successfully with DeepInfra")

            elif provider == "timelygpt":
                common_params = {
                    "api_key": TIMELYGPT_API_KEY,
                    "base_url": TIMELYGPT_API_BASE_URL,
                    "model": "openai/gpt-4.1-mini"
                }            
                self.llm = ChatOpenAI(**common_params, temperature=0)
                self.power_llm = ChatOpenAI(**common_params, temperature=0.5)
                self.json_llm = ChatOpenAI(**common_params, temperature=0.5, 
                    model_kwargs={"response_format": {"type": "json_object"}}
                )
                print("LLM initialized successfully with TIMELYGPT")

            # elif provider == "google":
            #     self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            #     self.power_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5)
            #     self.json_llm = ChatGoogleGenerativeAI(
            #         model="gemini-1.5-pro",
            #         temperature=0,
            #         generation_config={"response_mime_type": "application/json"}
            #     )
            # print("LLM initialized successfully")
        except Exception as e:
            print(f"Error initializing LLMs: {e}")
            raise e
        
    def get_model(self, model_tier="llm"):
        if model_tier == "power": return self.power_llm
        if model_tier == "json": return self.json_llm
        return self.llm