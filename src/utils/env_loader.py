import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_openai_api_key():
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found."
        )
    
    return api_key.strip()
