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
):
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
    user_prompt = config["prompt_template"].format(
        examples=examples_text, input_text=input_text
    )
    print("Hei")
    # try:
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": """
                    You are a helpful assistant that finds named entities (NER) in text.                         
                    """.strip(),
            },
            {"role": "user", "content": user_prompt},
        ],
        # response_format={
        #     "type": "json_schema",
        #     "json_schema": response_schema.model_json_schema(),
        # },
        # temperature=config["llm"]["temperature"],
        # max_tokens=config["llm"]["max_tokens"],
    )

    print("Type:", type(completion.choices[0].message.content))

    text = (completion.choices[0].message.content or "").strip()
    return text

    # except Exception as e:
    #     logging.info(f"OpenAI client generation failed: {e}")
    #     print("Error:", e)
    #     return ""


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
