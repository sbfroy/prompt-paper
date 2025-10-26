import logging
import os
import random
from pathlib import Path
from typing import Union

from openai import OpenAI
from pydantic import RootModel
from pypdf import PdfReader

from proptimize.data_manager import DataManager
from proptimize.schemas import InputExample, InputDataset
from proptimize.stages.client import get_llm_response

logging.basicConfig(level=logging.INFO)


class GeneratedExample(RootModel):
    """Schema for a single generated example."""
    root: dict[str, Union[str, dict[str, list[str]]]]


class GeneratedExamples(RootModel):
    """Schema for a batch of generated examples."""
    root: list[dict[str, Union[str, dict[str, list[str]]]]]


def read_pdfs_from_directory(path) -> str:
    """Read all PDFs from a directory and return concatenated text.

    Args:
        path: Path to a folder containing PDF files.

    Returns:
        Concatenated text from all PDFs in the folder.
    """
    path = Path(path)

    if not path.is_dir():
        raise ValueError(f"Path must be a directory: {path}")

    all_text = []
    pdf_files = sorted(path.glob("*.pdf"))  # Get all PDF files, sorted

    if not pdf_files:
        logging.warning(f"No PDF files found in {path}")
        return ""

    for pdf_file in pdf_files:
        logging.info(f"Reading PDF: {pdf_file.name}")
        try:
            reader = PdfReader(pdf_file)
            pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            all_text.append(pdf_text)
        except Exception as e:
            logging.error(f"Error reading {pdf_file.name}: {e}")
            continue

    return "\n".join(all_text)


def split_text_into_batches(text: str, n_batches: int) -> list[str]:
    """Split text into n_batches of equal length.

    Args:
        text: The full text to split.
        n_batches: Number of batches to create.

    Returns:
        List of text batches.
    """
    if not text:
        logging.warning("No text provided for batching")
        return []

    # Calculate batch size
    total_length = len(text)
    batch_size = total_length // n_batches

    batches = []

    for i in range(n_batches):
        start = i * batch_size
        if i == n_batches - 1:
            # Last batch gets all remaining text
            batches.append(text[start:])
        else:
            end = start + batch_size
            batches.append(text[start:end])

    return batches


class GenerateStage:
    def __init__(self, data_manager, config, client):
        self.data_manager = data_manager
        self.config = config
        self.client = client

    def run(self):
        logging.info("Starting generation stage...")

        scope_dir = self.config["scope_directory"]
        logging.info(f"Reading PDFs from scope directory: {scope_dir}")
        all_text = read_pdfs_from_directory(scope_dir)

        n_batches = self.config["target_num_examples"] // self.config["batch_size_examples"]
        logging.info(f"Splitting text into {n_batches} batches...")
        text_batches = split_text_into_batches(all_text, n_batches)

        logging.info(f"Created {len(text_batches)} text batches")

        all_examples = []
        base_temperature = self.config.get("temperature", 0.7)

        for i, text_batch in enumerate(text_batches):
            logging.info(f"Generating examples from batch {i+1}/{len(text_batches)}...")

            # Vary temperature slightly for diversity
            temperature = base_temperature + random.uniform(-0.1, 0.1)
            temperature = max(0.0, min(1.0, temperature))  # Clamp to [0, 1]

            batch_config = {
                "system_prompt": self.config["system_prompt"],
                "user_prompt": self.config["user_prompt"],
                "temperature": temperature,
            }

            try:
                response = get_llm_response(
                    config=batch_config,
                    client=self.client,
                    response_schema=GeneratedExamples,
                    is_generation=True,
                    text_batch=text_batch,
                    batch_size_examples=self.config["batch_size_examples"],
                )

                # Extract examples from response
                if response and isinstance(response, dict) and "root" in response:
                    batch_examples = response["root"]
                    all_examples.extend(batch_examples)
                    logging.info(f"Generated {len(batch_examples)} examples from batch {i+1}")
                else:
                    logging.warning(f"No valid examples generated from batch {i+1}")

            except Exception as e:
                logging.error(f"Error generating examples from batch {i+1}: {e}")
                continue

        logging.info(f"Total examples generated: {len(all_examples)}")

        # Convert to InputExample format and assign IDs
        input_examples = []
        for idx, example in enumerate(all_examples):
            input_text = example.get("input", "")
            output = example.get("output", "")

            # Convert output to string if it's a dict
            if isinstance(output, dict):
                output_str = str(output)
            else:
                output_str = str(output)

            input_examples.append(
                InputExample(
                    id=str(idx + 1),
                    input=input_text,
                    output=output_str
                )
            )

        # Create train/validation split (e.g., 80/20)
        random.shuffle(input_examples)
        split_idx = int(len(input_examples) * 0.8)

        train_examples = input_examples[:split_idx]
        val_examples = input_examples[split_idx:]

        # Save datasets
        train_dataset = InputDataset(examples=train_examples, task_type=self.data_manager.task)
        val_dataset = InputDataset(examples=val_examples, task_type=self.data_manager.task)

        train_path = self.data_manager.save_input_dataset(train_dataset, "generated_train.jsonl")
        val_path = self.data_manager.save_input_dataset(val_dataset, "generated_val.jsonl")

        logging.info(f"Saved {len(train_examples)} training examples to {train_path}")
        logging.info(f"Saved {len(val_examples)} validation examples to {val_path}")

        return train_path, val_path


def run_generate_stage(task, base_dir, config):
    # Setup
    data_manager = DataManager(task, base_dir)

    # TODO: Where should the client be initialized?

    # Run generation
    stage = GenerateStage(data_manager, config)
    return stage.run()

