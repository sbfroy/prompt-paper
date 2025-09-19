from vllm import LLM, SamplingParams
import logging
import time
import os

logging.basicConfig(level=logging.INFO)

# ====== LLM INTERACTION ======

def get_llm_response(prompt_template, individual, input_text, model, temperature, max_tokens):
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

    llm = LLM(model=model, dtype="half") # Use half precision for efficiency
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    max_retries=3 # Maximum number of retry attempts
    retry_delay=1 # Base delay between retries (exponential backoff)
    
    for attempt in range(max_retries):
        try:
            outputs = llm.generate([prompt], sampling_params=params)
            return outputs.outputs[0].text.strip()
            
        except RuntimeError as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logging.info(f"RuntimeError: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.info(f"Failed after {max_retries} attempts: {e}")
                return ""
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logging.info(f"Unexpected error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.info(f"Failed after {max_retries} attempts: {e}")
                return ""
