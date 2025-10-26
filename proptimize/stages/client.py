# %%
import json
import logging
import os
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, RootModel

from proptimize.schemas import InputExample

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_llm_response(
    config: dict,
    client: OpenAI,
    response_schema: BaseModel,
    is_generation: bool = False,
    individual: Optional[list[tuple[str, InputExample]]] = None,
    input_text: Optional[str] = None,
    text_batch: Optional[str] = None,
    batch_size_examples: Optional[int] = None,
) -> dict[str, list[str]]:
    """Generate a response from the LLM for either evaluation or generation.

    This function supports two modes:
    1. Evaluation mode (is_generation=False): Uses ICL examples to predict output for input_text
    2. Generation mode (is_generation=True): Generates new examples from text_batch

    Args:
        config: Config containing prompt templates and LLM params.
        client: OpenAI-compatible client (vLLM OpenAI server).
        response_schema: Pydantic model class for the expected response format.
        is_generation: If True, use generation mode. If False, use evaluation mode.
        individual: The ICL examples (required for evaluation mode).
        input_text: The input to evaluate (required for evaluation mode).
        text_batch: The text batch to generate examples from (required for generation mode).
        batch_size_examples: Number of examples to generate (required for generation mode).

    Returns:
        The model's response as a dictionary, or empty string on failure.
    """
    if is_generation:
        # Generation mode: generate new examples from text batch
        if text_batch is None or batch_size_examples is None:
            raise ValueError("text_batch and batch_size_examples are required for generation mode")

        user_prompt = config["user_prompt"].format(
            text_batch=text_batch,
            batch_size_examples=batch_size_examples
        )
    else:
        # Evaluation mode: predict output for input_text using ICL examples
        if individual is None or input_text is None:
            raise ValueError("individual and input_text are required for evaluation mode")

        if not isinstance(individual, list) or len(individual) == 0:
            raise ValueError("Individual must be a non-empty list")

        # Format examples as "Input: <input>\nOutput: <output>"
        formatted_examples = []
        for _, example in individual:
            formatted_example = f"Input: {example.input}\nOutput: {example.output}"
            formatted_examples.append(formatted_example)

        examples_text = "\n\n".join(formatted_examples)

        # Insert examples and test sentence into template
        user_prompt = config["user_prompt"].format(
            examples=examples_text, input_text=input_text
        )

    # Build messages list
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": user_prompt}
    ]

    # Call LLM
    completion = client.chat.completions.parse(
        model="openai/gpt-oss-120b",
        messages=messages,
        response_format=response_schema,
        extra_body=dict(guided_decoding_backend="outlines"),
        temperature=config.get("temperature", 0.0),
    )

    out = completion.choices[0].message.content

    if out is None:
        return ""

    try:
        return response_schema.model_validate_json(out).model_dump()
    except Exception as e:
        logging.info(f"OpenAI client generation failed: {e}")
        print("Warning:", e)
        return ""