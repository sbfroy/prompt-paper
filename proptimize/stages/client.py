# %%
import json
import os
from pydantic import RootModel, BaseModel
from openai import OpenAI

import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

from proptimize.schemas import InputExample


def get_llm_response(
    config: dict,
    client: OpenAI,
    individual: list[tuple[str, InputExample]],
    input_text: str,
    response_schema: BaseModel,
) -> dict[str, list[str]]:
    """
    Generate a response from the LLM using a prompt template and given input text.

    Args:
        config: Config containing prompt template and LLM params
        client: OpenAI-compatible client (vLLM OpenAI server)
        individual: The ICL examples
        input_text: The input to evaluate
        response_schema: Pydantic model class for the expected response format

    Returns:
        str: The model's response or empty string on failure
    """
    assert (
        isinstance(individual, list) and len(individual) > 0
    ), "Individual must be a non-empty list"

    examples_text = "\n\n".join(example.text for _, example in individual)

    # Insert examples and test sentence into template
    user_prompt = config["user_prompt"].format(
        examples=examples_text, input_text=input_text
    )

    # Build messages list
    messages = []

    # Add system and user messages
    messages.append({"role": "system", "content": config["system_prompt"]})
    messages.append({"role": "user", "content": user_prompt})

    # try:
    completion = client.chat.completions.parse(
        model="openai/gpt-oss-120b",
        messages=messages,
        response_format=response_schema,
        extra_body=dict(guided_decoding_backend="outlines"),
        # extra_body={"guided_json": response_schema.model_json_schema()},
        temperature=config["llm"]["temperature"],
        # max_tokens=config["llm"]["max_tokens"],
    )

    out = completion.choices[0].message.content

    if out is None:
        return ""

    try:
        return response_schema.model_validate_json(out).model_dump()
    except Exception as e:
        logging.info(f"OpenAI client generation failed: {e}")
        print("Warning:", e)


if __name__ == "__main__":
    from openai import OpenAI
    from pydantic import BaseModel, Field

    class ResponseSchema(BaseModel):
        ner_tags: list[str] = Field(
            description="List of named entity tags found in the text, using IOB2 format."
        )

    class XBRLResponse(RootModel[dict[str, list[str]]]):
        """A dictionary where keys are XBRL tags and values are lists of extracted strings."""

        pass

    client = OpenAI(
        base_url=f'http://localhost:{os.getenv("LLM_PORT", "8000")}/v1',
        api_key=os.getenv("LLM_API_KEY", "prompt-paper"),
    )

    res = get_llm_response(
        config={
            "prompt_template": """
                Extract the named entities in this text using 139 XBRL tags in the IOB2 format.
                
                Here are some examples:

                {examples}

                Return the results in JSON format. 
                Each key must be an XBRL tag. Each value must be a list of string values.
                If there are no entities in the text, return an empty JSON object.

                User: '{input_text}'
            
            """,
            "llm": {"temperature": 0.0, "max_tokens": 512},
        },
        client=client,
        individual=[("id", InputExample(text="text", example_id="id"))],
        input_text="Here is a sentence about Apple Inc. based in Cupertino, California.",
        response_schema=XBRLResponse,
    )

    print("res:", res)
