import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Set up API keys and environment variables
DEEPINFRA_API_KEY = os.getenv('DEEPINFRA_API_KEY')
DEEPINFRA_API_BASE_URL = "https://api.deepinfra.com/v1/openai"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

TIMELYGPT_API_KEY = os.getenv('TIMELYGPT_API_KEY')
TIMELYGPT_API_BASE_URL = "https://hello.timelygpt.co.kr/api/v2/chat/bridge/openai"