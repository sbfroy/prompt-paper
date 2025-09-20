import logging

logging.basicConfig(level=logging.INFO)

# ====== LLM INTERACTION ======

def get_llm_response(prompt_template, individual, input_text, llm_instance, sampling_params):
    """   
    Args:
        prompt_template: Prompt template with placeholders for individual and input text
        individual: The ICL examples from the GA
        input_text: The input to be tested
        llm_instance: Pre-initialized vLLM instance
        sampling_params: Pre-initialized sampling parameters
    
    Returns:
        str: Model response or empty string on failure
    """

    examples_text = "\n\n".join(example.text for _, example in individual)
    
    # Insert examples and test sentence into template
    prompt = prompt_template.format(
        examples=examples_text,
        input_text=input_text
    )

    try:
        res = llm_instance.generate([prompt], sampling_params=sampling_params)
        text = (res[0].outputs[0].text or "").strip() # falls back to empty string
        return text

    except Exception as e:
        logging.info(f"vLLM generation failed: {e}")
        return ""