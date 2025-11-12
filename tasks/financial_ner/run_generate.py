"""
Financial NER - Generation Stage

This script runs the generation stage for financial NER, which:
1. Uses an LLM to generate synthetic training examples
2. Creates both positive (with XBRL entities) and negative (without) examples
3. Validates generated examples match expected format
"""

import sys
import logging
import os
from pathlib import Path
from typing import Union, Literal

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import RootModel, Field, BaseModel

from grasp.wandb_utils import init_wandb, finish_wandb
from grasp.stages.generate.main import run_generate_stage
from grasp.run_vllm import start_vllm_servers

load_dotenv()

logging.basicConfig(level=logging.INFO)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))  # step out to 'GRaSp'


# ============================================================================
# Task-Specific Schema Definitions
# ============================================================================


class GeneratedExample(BaseModel):
    """Schema for a single generated financial NER example."""

    input: str = Field(..., description="Financial sentence")
    output: Union[dict[str, list[str]], Literal["No XBRL associated data."]] = Field(
        ..., description="XBRL tags mapping or 'No XBRL associated data.'"
    )


class GeneratedExamples(RootModel):
    """Schema for a batch of generated financial NER examples."""

    root: list[GeneratedExample]


# ============================================================================
# Task-Specific Validation Function
# ============================================================================


def validate_financial_ner_examples(
    examples: list[dict], expected_type: Literal["positive", "negative"]
) -> list[dict]:
    """
    Validate financial NER examples match expected type and format.

    Args:
        examples: Raw examples from LLM
        expected_type: "positive" (with XBRL entities) or "negative" (without)

    Returns:
        Filtered list of valid examples
    """
    validated = []

    for example in examples:
        try:
            output = example.get("output")

            if expected_type == "positive":
                # Must be a dict with at least one XBRL tag
                if isinstance(output, dict) and len(output) > 0:
                    validated.append(example)
                else:
                    logging.debug(f"Rejected positive example: {output}")

            elif expected_type == "negative":
                # Must be the exact string "No XBRL associated data."
                if output == "No XBRL associated data.":
                    validated.append(example)
                else:
                    logging.debug(f"Rejected negative example: {output}")

        except Exception as e:
            logging.warning(f"Error validating example: {e}")
            continue

    return validated


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Run the generation stage for financial NER."""
    config = load_config()

    # Set up paths
    task_dir = Path(__file__).parent
    base_dir = task_dir.parent

    # Initialize wandb for experiment tracking
    init_wandb(task_name=config["task"], config=config)

    # Initialize OpenAI-compatible client
    logging.info("Creating OpenAI client for generation...")

    client = OpenAI(
        base_url=f'http://localhost:{os.getenv("LLM_PORT", "8000")}/v1',
        api_key=os.getenv("LLM_API_KEY", "prompt-paper"),
    )

    # Run generation with task-specific schema and validation
    run_generate_stage(
        task=config["task"],
        base_dir=str(base_dir),
        config={**config["generation"], "dataset_size": config["dataset"]["size"]},
        client=client,
        response_schema=GeneratedExamples,
        validation_fn=validate_financial_ner_examples,
    )

    finish_wandb()

    logging.info("Generation stage completed successfully!")


if __name__ == "__main__":
    start_vllm_servers(start_LLM=True, start_embedding=False)
    main()
