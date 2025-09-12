from openai import OpenAI
from openai import BadRequestError, RateLimitError
from dotenv import load_dotenv
import time
import os

MODEL_PRICING = {
    # per 1M tokens in NOK
    "gpt-4o-mini": {"input": 1.65, "output": 6.60},  
}

class _CostTracker:
    """
    Tracks the cost and token usage of all API calls.
    Keeps a running sum of:
        - total cost 
        - total prompt tokens used
        - total completion tokens used
    """
    def __init__(self):
        self._total_cost = 0.0
        self._total_tokens = {"prompt": 0, "completion": 0}

    def add_usage(self, usage, model):
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        cost = calculate_cost(prompt_tokens, completion_tokens, model)
       
        self._total_cost += cost
        self._total_tokens["prompt"] += prompt_tokens
        self._total_tokens["completion"] += completion_tokens

    def get_total_cost(self):
        return self._total_cost

    def get_total_tokens(self):
        return dict(self._total_tokens)

    def reset_all(self):
        self._total_cost = 0.0
        self._total_tokens = {"prompt": 0, "completion": 0}

_tracker = _CostTracker()

# ====== CLIENT INIT ======

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ====== COST CALCULATION ======

def calculate_cost(prompt_tokens, completion_tokens, model):
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost

# ====== PUBLIC API ======

def get_total_cost():
    return _tracker.get_total_cost()

def get_total_tokens():
    return _tracker.get_total_tokens()

def reset_cost_tracking():
    _tracker.reset_all()

# ====== LLM INTERACTION ======

def get_llm_response(prompt_template, individual, input_text, model):
    """   
    Args:
        prompt_template: Prompt template with placeholders for individual and input text
        individual: 
        input_text: 
        model: 
    
    Returns:
        str: Response from the LLM
    """

    examples_text = "\n\n".join(example.text for _, example in individual)
    
    # Insert examples and test sentence into template
    prompt = prompt_template.format(
        examples=examples_text,
        input_text=input_text
    )

    max_retries=3 # Maximum number of retry attempts
    retry_delay=1 # Base delay between retries (exponential backoff)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            _tracker.add_usage(response.usage, model)
            
            return response.choices[0].message.content
            
        except BadRequestError as e:
            print(f"BadRequestError: {e}. Skipping this prompt.")
            return ""  # Return empty string instead of crashing
        except ValueError as e:
            print(f"ValueError (possibly content filter): {e}. Skipping this prompt.")
            return ""
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Rate limit error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Rate limit error after {max_retries} attempts: {e}")
                return ""
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Unexpected error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return ""
