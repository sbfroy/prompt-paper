import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)

def get_llm_response(config, client, individual, input_text):
    """
    Generate a response from the LLM using a prompt template and given input text.

    Args:
        config: Config containing prompt template and LLM params
        client: OpenAI-compatible client (vLLM OpenAI server)
        individual: The ICL examples
        input_text: The input to evaluate

    Returns:
        str: The model's response or empty string on failure
    """

    examples_text = "\n\n".join(example.text for _, example in individual)

    # Insert examples and test sentence into template
    user_prompt = config['prompt_template'].format(
        examples=examples_text,
        input_text=input_text
    )

    try:
        completion = client.chat.completions.create(
            model=config['llm']['model'],
            messages=[
                {'role': 'user', 'content': user_prompt}
            ],
            temperature=config['llm']['temperature'],
            max_tokens=config['llm']['max_tokens'],
        )
        text = (completion.choices[0].message.content or "").strip()
        return text

    except Exception as e:
        logging.info(f"OpenAI client generation failed: {e}")
        return ""