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
        validation_dataset = self.data_manager.load_input_dataset(
            "val", 
            dataset_size=8000,
            use_real=config["use_real"]
        )
        
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

        # Load test dataset from wandb artifact
        test_dataset = self.data_manager.load_input_dataset(
            "test",
            dataset_size=8000,
            use_real=config["use_real"]
        )
        
        # Convert to dictionary format for compatibility
        self.test_data = [
            {"input": example.input, "output": example.output}
            for example in test_dataset.examples
        ]
        
        logging.info(f"Loaded {len(self.test_data)} test examples")

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
            Dictionary with micro_precision, micro_recall, micro_f1, macro_f1
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

        # Aggregate TP/FP/FN across all labels and examples
        all_label_stats = {}  # {label: {"tp": int, "fp": int, "fn": int}}
        checkpoint_label_stats = {}  # For early stopping checkpoint

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
                    "Response is None, treating as empty prediction. Something may be wrong."
                )
                response = {}

            # Compute per-label stats for this example and aggregate
            example_stats = self.compute_example_stats(gold_labels, response)
            for label, stats in example_stats.items():
                if label not in all_label_stats:
                    all_label_stats[label] = {"tp": 0, "fp": 0, "fn": 0}
                all_label_stats[label]["tp"] += stats["tp"]
                all_label_stats[label]["fp"] += stats["fp"]
                all_label_stats[label]["fn"] += stats["fn"]

            # Check for early stopping after reaching the checkpoint
            if (self.prev_gen_avg is not None and
                    idx + 1 == early_stop_checkpoint):

                # Compute micro-F1 at checkpoint for early stopping decision
                checkpoint_metrics = self.compute_metrics_from_stats(all_label_stats)
                current_micro_f1 = checkpoint_metrics["micro_f1"]
                threshold = self.prev_gen_avg - (std_multiplier * self.prev_gen_std)

                if current_micro_f1 < threshold:
                    # Individual is underperforming, stop probabilistically
                    early_stop_prob = self.config["early_stop_probability"]
                    if random.random() < early_stop_prob:
                        self.early_stopped_count += 1
                        return checkpoint_metrics
                    # Otherwise, continue evaluating this individual

        # Compute final metrics
        if not all_label_stats:
            return {
                "micro_precision": 0.0,
                "micro_recall": 0.0,
                "micro_f1": 0.0,
                "macro_f1": 0.0,
            }

        return self.compute_metrics_from_stats(all_label_stats)

    def evaluate_on_test_set(self, individual):
        """
        Evaluate an individual on the test set without early stopping.
        
        Args:
            individual: List of (cluster_id, example) tuples

        Returns:
            Dictionary with micro_f1 and macro_f1
        """
        # Aggregate TP/FP/FN across all labels and examples
        all_label_stats = {}  # {label: {"tp": int, "fp": int, "fn": int}}

        for example in self.test_data:
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
                logging.warning(
                    "Response is None during test evaluation, treating as empty prediction"
                )
                response = {}

            # Compute per-label stats for this example and aggregate
            example_stats = self.compute_example_stats(gold_labels, response)
            for label, stats in example_stats.items():
                if label not in all_label_stats:
                    all_label_stats[label] = {"tp": 0, "fp": 0, "fn": 0}
                all_label_stats[label]["tp"] += stats["tp"]
                all_label_stats[label]["fp"] += stats["fp"]
                all_label_stats[label]["fn"] += stats["fn"]

        # Compute final metrics
        if not all_label_stats:
            metrics = {"micro_f1": 0.0, "macro_f1": 0.0}
        else:
            full_metrics = self.compute_metrics_from_stats(all_label_stats)
            # Only return micro_f1 and macro_f1 for test set
            metrics = {
                "micro_f1": full_metrics["micro_f1"],
                "macro_f1": full_metrics["macro_f1"],
            }

        logging.info(
            f"Test set evaluation complete: micro_f1={metrics['micro_f1']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
        return metrics

    @staticmethod
    def compute_example_stats(
        gold: dict[str, list[str]], pred: dict[str, list[str]]
    ) -> dict[str, dict[str, int]]:
        """
        Compute per-label TP/FP/FN statistics for a single example.

        Args:
            gold: Dictionary containing the gold labels {label: [values]}
            pred: Dictionary containing the predicted labels {label: [values]}

        Returns:
            Dictionary mapping label names to {"tp": int, "fp": int, "fn": int}
        """
        label_stats = {}
        all_labels = set(gold.keys()) | set(pred.keys())

        for label in all_labels:
            gold_values = set(gold.get(label, []))
            pred_values = set(pred.get(label, []))

            tp = len(gold_values & pred_values)
            fp = len(pred_values - gold_values)
            fn = len(gold_values - pred_values)

            label_stats[label] = {"tp": tp, "fp": fp, "fn": fn}

        return label_stats

    @staticmethod
    def compute_metrics_from_stats(
        all_label_stats: dict[str, dict[str, int]]
    ) -> dict[str, float]:
        """
        Compute micro and macro F1 metrics from aggregated label statistics.

        Args:
            all_label_stats: Dictionary mapping label names to aggregated {"tp", "fp", "fn"}

        Returns:
            Dictionary with micro_precision, micro_recall, micro_f1, macro_f1
        """
        # Micro metrics: sum TP/FP/FN across all labels
        total_tp = sum(stats["tp"] for stats in all_label_stats.values())
        total_fp = sum(stats["fp"] for stats in all_label_stats.values())
        total_fn = sum(stats["fn"] for stats in all_label_stats.values())

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        # Macro F1: compute F1 per label, then average
        label_f1s = []
        for label, stats in all_label_stats.items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            label_f1s.append(f1)

        macro_f1 = sum(label_f1s) / len(label_f1s) if label_f1s else 0.0

        return {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        }


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
        Tuple of (validation_eval_fn, test_eval_fn)
    """
    evaluator = Evaluator(data_manager, eval_config, client)
    return evaluator.evaluate_individual, evaluator.evaluate_on_test_set


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

    eval_fn, test_eval_fn = create_evaluator(
        data_manager=data_manager,
        eval_config={
            **config["evaluation"], 
            "dataset_size": config["dataset"]["size"],
            "use_real": config["dataset"]["use_real"]
        },
        client=client
    )

    run_evolve_stage(
        task=config["task"],
        base_dir=str(base_dir),
        config={
            **config["evolution"], 
            "dataset_size": config["dataset"]["size"],
            "use_real": config["dataset"]["use_real"]
        },
        eval_fn=eval_fn,
        test_eval_fn=test_eval_fn,
    )

    finish_wandb()

    logging.info("Evolution stage completed successfully!")


if __name__ == "__main__":
    start_vllm_servers(start_embedding=False, start_LLM=True)
    main()
