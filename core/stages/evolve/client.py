from openai import OpenAI
from openai import BadRequestError, RateLimitError
from dotenv import load_dotenv
import os
import time

# Global cost tracking
_total_cost = 0.0
_total_tokens = {"prompt": 0, "completion": 0}
_generation_cost = 0.0
_generation_tokens = {"prompt": 0, "completion": 0}
_generation_calls = 0

MODEL_PRICING = {
    "gpt-4o-mini": {"input": 1.65, "output": 6.60},  # per 1M tokens in NOK
}

def calculate_cost(usage, model):
    """Calculate cost based on token usage and model pricing."""
    if model not in MODEL_PRICING:
        print(f"Warning: Unknown model {model}, cost tracking unavailable")
        return 0.0
    
    pricing = MODEL_PRICING[model]
    input_cost = (usage.prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (usage.completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost

def get_total_cost():
    # Total API cost so far 
    return _total_cost

def get_total_tokens():
    # Total token usage so far
    return _total_tokens.copy()

def reset_cost_tracking():
    """Reset cost tracking (useful for new experiments)."""
    global _total_cost, _total_tokens, _generation_cost, _generation_tokens, _generation_calls
    _total_cost = 0.0
    _total_tokens = {"prompt": 0, "completion": 0}
    _generation_cost = 0.0
    _generation_tokens = {"prompt": 0, "completion": 0}
    _generation_calls = 0

def print_generation_cost_summary():
    """Get generation cost summary for inline display."""
    global _generation_cost, _generation_tokens, _generation_calls
    if _generation_calls > 0:
        gen_cost = _generation_cost
        total_cost = _total_cost
        
        # Reset generation counters
        _generation_cost = 0.0
        _generation_tokens = {"prompt": 0, "completion": 0}
        _generation_calls = 0
        
        return f"(Gen: {gen_cost:.3f} NOK | Total: {total_cost:.3f} NOK)"
    return ""

def get_llm_response(prompt_template, individual, test_sentence, cluster_dataset, model="gpt-4o-mini"):
    """   
    Args:
        prompt_template: String template with placeholders for examples
        individual: List of (cluster_id, example) pairs from GA
        test_sentence: The sentence to evaluate
        cluster_dataset: The dataset containing all clusters to lookup example text
        model: OpenAI model to use
    
    Returns:
        str: Raw response from the LLM
    """
    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    examples_text = "\n\n".join(example.text for _, example in individual)
    
    # Insert examples and test sentence into template
    prompt = prompt_template.format(
        examples=examples_text,
        test_sentence=test_sentence
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
            
            # Track cost and tokens
            global _total_cost, _total_tokens, _generation_cost, _generation_tokens, _generation_calls
            if hasattr(response, 'usage') and response.usage:
                cost = calculate_cost(response.usage, model)
                
                # Update totals
                _total_cost += cost
                _total_tokens["prompt"] += response.usage.prompt_tokens
                _total_tokens["completion"] += response.usage.completion_tokens
                
                # Update generation tracking
                _generation_cost += cost
                _generation_tokens["prompt"] += response.usage.prompt_tokens
                _generation_tokens["completion"] += response.usage.completion_tokens
                _generation_calls += 1
            
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
