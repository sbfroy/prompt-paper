# %%
import logging
from typing import Optional, Union

from openai import OpenAI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_llm_response(
    client: OpenAI,
    messages: list[dict[str, str]],
    response_schema: Optional[BaseModel] = None,
    temperature: float = 0.0,
    model: str = "openai/gpt-oss-120b",
) -> Optional[Union[dict, str]]:
    """Generate a response from the LLM using the provided messages.

    Args:
        client: OpenAI-compatible client (vLLM OpenAI server).
        messages: List of message dicts with 'role' and 'content' keys.
        response_schema: Optional Pydantic model class for structured output.
                        If None, returns raw text response.
        temperature: Sampling temperature for the LLM.
        model: Model name to use for generation.

    Returns:
        If response_schema is provided: dict representation of the response.
        If response_schema is None: raw text string.
        Returns None on failure.
    """
    try:
        if response_schema is not None:
            # Structured output mode
            completion = client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_schema,
                extra_body=dict(guided_decoding_backend="outlines"),
                temperature=temperature,
            )
            
            out = completion.choices[0].message.content
            
            if out is None:
                return None
            
            return response_schema.model_validate_json(out).model_dump()
        else:
            # Plain text mode
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            
            return completion.choices[0].message.content
            
    except Exception as e:
        logging.error(f"LLM client generation failed: {e}")
        return None