"""
Generate stage for creating synthetic training data from PDF documents.

This module handles:
- Reading and extracting text from PDF documents
- Generating positive and negative examples using LLMs
- Splitting data into train/validation sets
"""

import logging
import random
import json
from pathlib import Path
from typing import Literal, Callable, Optional

from pydantic import RootModel
from pypdf import PdfReader
from tqdm import tqdm

from grasp.data_manager import DataManager
from grasp.schemas import InputExample, InputDataset
from grasp.stages.client import get_llm_response

logging.basicConfig(level=logging.INFO)



def read_pdfs_from_directory(path) -> str:
    """
    Read all PDFs from a directory and return concatenated text.

    Args:
        path: Path to a folder containing PDF files.

    Returns:
        Concatenated text from all PDFs in the folder.
    """
    path = Path(path)

    if not path.is_dir():
        raise ValueError(f"Path must be a directory: {path}")

    all_text = []
    pdf_files = sorted(path.glob("*.pdf"))

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


def sample_random_chunks(text: str, n_samples: int, chunk_size: int) -> list[str]:
    """
    Sample random chunks from text for diversity.

    Args:
        text: The full text to sample from.
        n_samples: Number of random chunks to sample.
        chunk_size: Size of each chunk in characters.

    Returns:
        List of random text chunks.
    """
    if not text or len(text) < chunk_size:
        logging.warning("Text too short for chunking")
        return [text] if text else []

    chunks = []
    max_start = len(text) - chunk_size

    for _ in range(n_samples):
        start = random.randint(0, max_start)
        chunks.append(text[start:start + chunk_size])

    return chunks


class GenerateStage:
    """
    Main class for generating synthetic training data.

    This stage reads PDF documents, generates both positive and negative examples
    using an LLM, validates them, and creates train/validation splits.
    """

    def __init__(
        self,
        data_manager: DataManager,
        config: dict,
        client,
        response_schema: RootModel,
        validation_fn: Optional[Callable[[list[dict], Literal["positive", "negative"]], list[dict]]] = None
    ):
        """
        Initialize the generate stage.

        Args:
            data_manager: DataManager instance for the task.
            config: Generation configuration from YAML.
            client: LLM client for generation.
            response_schema: Pydantic schema for LLM response validation.
            validation_fn: Optional function to validate examples by type.
                          Should accept (examples, expected_type) and return filtered examples.
                          If None, no validation is performed.
        """
        self.data_manager = data_manager
        self.config = config
        self.client = client
        self.response_schema = response_schema
        self.validation_fn = validation_fn

    def run(self):
        """
        Execute the generation stage.

        Returns:
            Tuple of (train_artifact, val_artifact) for generated datasets.
        """
        logging.info("Starting generation stage...")

        # Read source documents
        scope_dir = self.data_manager.get_scope_dir()
        logging.info(f"Reading PDFs from scope directory: {scope_dir}")
        all_text = read_pdfs_from_directory(scope_dir)

        # Get target dataset size from config
        target_examples = self.config["dataset_size"]
        batch_size = self.config["batch_size_examples"]
        chunk_size = self.config["chunk_size"]

        target_positive = target_examples // 2
        target_negative = target_examples - target_positive

        positive_batches = target_positive // batch_size
        negative_batches = target_negative // batch_size

        logging.info(f"Generating {target_positive} positive examples in {positive_batches} batches")
        logging.info(f"Generating {target_negative} negative examples in {negative_batches} batches")

        all_examples = []

        # Generate positive examples (with entities)
        all_examples.extend(
            self._generate_examples(
                all_text,
                positive_batches,
                batch_size,
                chunk_size,
                example_type="positive"
            )
        )

        # Generate negative examples (without entities)
        all_examples.extend(
            self._generate_examples(
                all_text,
                negative_batches,
                batch_size,
                chunk_size,
                example_type="negative"
            )
        )

        logging.info(f"Total examples generated: {len(all_examples)}")

        # Convert to InputExample format
        input_examples = self._convert_to_input_examples(all_examples)

        # Create train/validation split (80/20)
        random.shuffle(input_examples)
        split_idx = int(len(input_examples) * 0.8)

        train_examples = input_examples[:split_idx]
        val_examples = input_examples[split_idx:]

        # Save datasets to WandB with size in artifact name
        train_dataset = InputDataset(examples=train_examples, task_type=self.data_manager.task)
        val_dataset = InputDataset(examples=val_examples, task_type=self.data_manager.task)

        dataset_size = target_examples
        train_artifact = self.data_manager.save_input_dataset(train_dataset, "train", dataset_size=dataset_size)
        val_artifact = self.data_manager.save_input_dataset(val_dataset, "val", dataset_size=dataset_size)

        logging.info(f"Saved {len(train_examples)} training examples to WandB artifact: {train_artifact.name}")
        logging.info(f"Saved {len(val_examples)} validation examples to WandB artifact: {val_artifact.name}")

        return train_artifact, val_artifact

    def _generate_examples(
        self,
        text: str,
        n_batches: int,
        batch_size: int,
        chunk_size: int,
        example_type: Literal["positive", "negative"]
    ) -> list[dict]:
        """
        Generate examples of a specific type using the LLM.
        
        Continues generating batches until target number is reached.
        Includes safety limit to prevent infinite loops if validation rate is too low.

        Args:
            text: Source text for generation.
            n_batches: Number of batches to generate initially (determines target count).
            batch_size: Examples per batch.
            chunk_size: Size of random text chunks.
            example_type: "positive" (with entities) or "negative" (without).

        Returns:
            List of generated examples.
        """
        target_count = n_batches * batch_size
        examples = []
        base_temperature = self.config["base_temperature"]
        
        # Track statistics
        total_generated = 0
        total_validated = 0
        batches_attempted = 0

        # Safety limit: allow up to 2x the original batches before giving up
        max_batches = n_batches * 2
        
        logging.info(f"Target: {target_count} {example_type} examples (will generate until reached)")

        desc = f"Generating {example_type} examples"
        pbar = tqdm(total=target_count, desc=desc, ncols=75)

        while len(examples) < target_count and batches_attempted < max_batches:
            batches_attempted += 1
            
            # Sample a random chunk for each batch
            text_chunk = sample_random_chunks(text, 1, chunk_size)[0]
            
            # Vary temperature for diversity
            temperature = base_temperature + random.uniform(-0.2, 0.2)

            # Get prompt from config based on example type
            if example_type == "positive":
                user_prompt = self.config["positive_user_prompt"].format(
                    text_chunk=text_chunk,
                    batch_size_examples=batch_size
                )
            else:
                user_prompt = self.config["negative_user_prompt"].format(
                    text_chunk=text_chunk,
                    batch_size_examples=batch_size
                )

            messages = [
                {"role": "system", "content": self.config["system_prompt"]},
                {"role": "user", "content": user_prompt}
            ]

            try:
                response = get_llm_response(
                    client=self.client,
                    messages=messages,
                    response_schema=self.response_schema,
                    temperature=temperature,
                )

                if response:
                    # response is already unwrapped by model_dump() in get_llm_response
                    # For RootModel[list[T]], it returns a list directly
                    batch_examples = response
                    total_generated += len(batch_examples)

                    # Validate examples if validation function provided
                    if self.validation_fn:
                        validated = self.validation_fn(batch_examples, example_type)
                    else:
                        validated = batch_examples

                    total_validated += len(validated)
                    examples.extend(validated)
                    
                    # Update progress bar
                    pbar.update(len(validated))

                else:
                    logging.warning(f"No valid response from batch {batches_attempted}")

            except Exception as e:
                logging.error(f"Error generating batch {batches_attempted}: {e}")
                continue

        pbar.close()
        
        # Calculate statistics
        validation_rate = (total_validated / total_generated * 100) if total_generated > 0 else 0
        
        logging.info(
            f"Generated {len(examples)}/{target_count} {example_type} examples "
            f"({batches_attempted} batches, {validation_rate:.1f}% validation rate)"
        )
        
        # Check if we hit the safety limit
        if batches_attempted >= max_batches and len(examples) < target_count:
            logging.warning(
                f"Hit safety limit ({max_batches} batches) with only {len(examples)}/{target_count} examples. "
                f"Validation rate is too low ({validation_rate:.1f}%). Check your validation logic and prompts."
            )
        
        return examples

    def _convert_to_input_examples(self, examples: list[dict]) -> list[InputExample]:
        """
        Convert raw examples to InputExample schema.

        Args:
            examples: List of raw examples from LLM.

        Returns:
            List of InputExample objects.
        """
        input_examples = []

        for idx, example in enumerate(examples):
            input_text = example.get("input", "")
            output = example.get("output", "")

            # Convert output to proper JSON string format
            if isinstance(output, dict):
                output_str = json.dumps(output)
            else:
                output_str = str(output)

            input_examples.append(
                InputExample(
                    id=str(idx + 1),
                    input=input_text,
                    output=output_str
                )
            )

        return input_examples


def run_generate_stage(
    task: str,
    base_dir: str,
    config: dict,
    client,
    response_schema: RootModel,
    validation_fn: Optional[Callable[[list[dict], Literal["positive", "negative"]], list[dict]]] = None
):
    """
    Run the generation stage to create synthetic data.

    Args:
        task: Task name.
        base_dir: Base directory containing task folders.
        config: Generation configuration from YAML.
        client: LLM client for generation.
        response_schema: Pydantic schema for LLM response validation.
        validation_fn: Optional function to validate examples by type.

    Returns:
        Tuple of (train_artifact, val_artifact) for generated datasets.
    """
    data_manager = DataManager(task, base_dir)
    stage = GenerateStage(data_manager, config, client, response_schema, validation_fn)
    return stage.run()
