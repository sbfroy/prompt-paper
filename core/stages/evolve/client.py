from openai import OpenAI
import os

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
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    examples_text = "\n\n".join(example.text for _, example in individual)
    
    # Insert examples and test sentence into template
    prompt = prompt_template.format(
        examples=examples_text,
        test_sentence=test_sentence
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content
