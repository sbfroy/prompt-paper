"""
Financial NER - Evolution Stage

This script runs the evolutionary algorithm stage for financial NER, which:
1. Uses a genetic algorithm to evolve in-context learning examples
2. Evaluates individuals on the validation set
3. Optimizes for F1 score on XBRL entity extraction
"""

import sys
import logging
import json
import os
import random
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import RootModel

from grasp.wandb_utils import init_wandb, finish_wandb
from grasp.stages.evolve import run_evolve_stage
from grasp.stages.client import get_llm_response
from grasp.run_vllm import start_vllm_servers
from grasp.data_manager import DataManager

load_dotenv()

logging.basicConfig(level=logging.INFO)

# Suppress verbose logging from external libraries
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

random.seed(42)


class XBRLResponse(RootModel[dict[str, list[str]]]):
    """Pydantic model for XBRL entity extraction responses."""
    pass


class Evaluator:
    """Evaluates ICL examples on the financial NER validation set."""

    def __init__(self, data_manager, config, client):
        """
        Initialize the evaluator.

        Args:
            data_manager: DataManager instance for loading validation data
            config: Evaluation configuration dictionary
            client: OpenAI-compatible client for LLM calls
        """
        self.data_manager = data_manager
        self.config = config
        self.client = client

        # Load validation dataset from wandb artifact
        validation_dataset = self.data_manager.load_input_dataset("val", dataset_size=10000) # config["dataset_size"]
        
        # Convert to dictionary format for compatibility
        self.validation_data = [
            {"input": example.input, "output": example.output}
            for example in validation_dataset.examples
        ]

        # Sample validation data based on ratio
        sample_size = int(
            len(self.validation_data) * self.config["validation_sample_ratio"]
        )
        self.validation_data = random.sample(self.validation_data, sample_size)

        # Statistics for early stopping of individuals
        self.prev_gen_avg = None
        self.prev_gen_std = None
        self.early_stopped_count = 0

    def update_generation_stats(self, avg, std):
        """
        Update statistics from the previous generation.

        Args:
            avg: Average fitness of previous generation
            std: Standard deviation of fitness of previous generation
        """
        self.prev_gen_avg = avg
        self.prev_gen_std = std

        if self.early_stopped_count > 0:
            logging.info(f"{self.early_stopped_count} individuals were early-stopped.")
        else:
            logging.info("No individuals were early-stopped.")

        self.early_stopped_count = 0

    def evaluate_individual(self, individual):
        """
        Evaluate an individual on the validation set.
        Data is shuffled independently for each individual to avoid bias.

        Args:
            individual: List of (cluster_id, example) tuples

        Returns:
            Average F1 score across validation set
        """
        # Shuffle validation data for this individual
        shuffled_validation_data = random.sample(
            self.validation_data, len(self.validation_data)
        )

        # Calculate early stopping checkpoint based on fraction of val set
        early_stop_fraction = self.config["early_stop_checkpoint_fraction"]
        early_stop_checkpoint = int(
            len(shuffled_validation_data) * early_stop_fraction
        )

        std_multiplier = self.config["early_stop_std_multiplier"]

        scores = []

        for idx, example in enumerate(shuffled_validation_data):
            user_input = example["input"]
            gold_output = example["output"]

            # Format ICL examples from individual
            formatted_examples = []
            for _, icl_example in individual:
                formatted_example = (
                    f"Input: {icl_example.input}\n"
                    f"Output: {icl_example.output}"
                )
                formatted_examples.append(formatted_example)

            examples_text = "\n\n".join(formatted_examples)

            # Construct messages for evaluation
            user_prompt = self.config["user_prompt"].format(
                examples=examples_text,
                input_text=user_input
            )

            messages = [
                {"role": "system", "content": self.config["system_prompt"]},
                {"role": "user", "content": user_prompt}
            ]

            response = get_llm_response(
                client=self.client,
                messages=messages,
                response_schema=XBRLResponse,
                temperature=self.config["temperature"]
            )

            # Convert gold_output to comparable dict
            if gold_output == "No XBRL associated data.":
                gold_labels = {}
            else:
                gold_labels = json.loads(gold_output.replace("'", '"'))

            if response is None:
                logging.info(
                    "Response is None, using score 0.0. Something may be wrong."
                )
                scores.append(0.0)
            else:
                # response is already unwrapped by model_dump() in get_llm_response
                # For RootModel[dict[str, list[str]]], it returns a dict directly
                score = self.compare_json_objects(gold_labels, response)
                scores.append(score)

            # Check for early stopping after reaching the checkpoint
            if (self.prev_gen_avg is not None and
                    idx + 1 == early_stop_checkpoint):

                current_avg = sum(scores) / len(scores)
                threshold = self.prev_gen_avg - (std_multiplier * self.prev_gen_std)

                if current_avg < threshold:
                    # Individual is underperforming, stop probabilistically
                    early_stop_prob = self.config["early_stop_probability"]
                    if random.random() < early_stop_prob:
                        self.early_stopped_count += 1
                        return current_avg
                    # Otherwise, continue evaluating this individual

        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def compare_json_objects(
        gold: dict[str, list[str]], pred: dict[str, list[str]]
    ) -> float:
        """
        Compare two JSON objects and calculate the F1 score.

        Args:
            gold: Dictionary containing the gold labels
            pred: Dictionary containing the predicted labels

        Returns:
            F1 score between gold and predicted labels
        """
        if not gold and not pred:
            return 1.0  # Perfect match if both are empty

        if not gold or not pred:
            return 0.0

        tp = 0
        fp = 0
        fn = 0

        for key, gold_values in gold.items():
            pred_values = pred.get(key, [])  # Default to empty list if key not found

            for value in gold_values:
                if value in pred_values:
                    tp += 1  # Correctly predicted
                else:
                    fn += 1  # In gold but not predicted

            # Predicted but not in gold
            fp += len(set(pred_values) - set(gold_values))

        # Keys in pred but not in gold
        # Keys in pred but not in gold
        extra_keys = set(pred.keys()) - set(gold.keys())

        for key in extra_keys:
            # All values in these keys are incorrect
            pred_values = pred[key]
            fp += len(pred_values)

        # Calculate precision, recall, and F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return f1


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_evaluator(data_manager, eval_config, client):
    """
    Create an evaluation function for the genetic algorithm.
    
    The GA requires a single-argument function, so this factory
    wraps the evaluator with the necessary configuration.

    Args:
        data_manager: DataManager instance for loading validation data
        eval_config: Evaluation configuration dictionary
        client: OpenAI-compatible client for LLM calls

    Returns:
        Function that evaluates individuals
    """
    evaluator = Evaluator(data_manager, eval_config, client)
    return evaluator.evaluate_individual


def main():
    """Run the evolution stage for financial NER."""
    config = load_config()

    # Set up paths
    task_dir = Path(__file__).parent
    base_dir = task_dir.parent

    # Initialize wandb for experiment tracking
    init_wandb(task_name=config["task"], config=config)

    # Initialize OpenAI-compatible client for evaluation
    logging.info("Creating OpenAI client for evaluation...")

    client = OpenAI(
        base_url=f'http://localhost:{os.getenv("LLM_PORT", "8000")}/v1',
        api_key=os.getenv("LLM_API_KEY", "prompt-paper"),
    )

    data_manager = DataManager(config["task"], str(base_dir))

    eval_fn = create_evaluator(
        data_manager=data_manager,
        eval_config={**config["evaluation"], "dataset_size": config["dataset"]["size"]},
        client=client
    )

    run_evolve_stage(
        task=config["task"],
        base_dir=str(base_dir),
        config={**config["evolution"], "dataset_size": config["dataset"]["size"]},
        eval_fn=eval_fn,
    )

    finish_wandb()

    logging.info("Evolution stage completed successfully!")


if __name__ == "__main__":
    start_vllm_servers(start_embedding=False, start_LLM=True)
    main()
