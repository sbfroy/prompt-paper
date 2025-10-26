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

from proptimize.wandb_utils import init_wandb, finish_wandb
from proptimize.stages.cluster import run_cluster_stage
from proptimize.stages.evolve import run_evolve_stage
from proptimize.stages.client import get_llm_response
from proptimize.run_vllm import start_vllm_servers

load_dotenv()  

logging.basicConfig(level=logging.INFO)

# Some logging suppression
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))  # step out to 'prompt-paper'

random.seed(42)


class XBRLResponse(RootModel[dict[str, list[str]]]):
    pass


class Evaluator:
    def __init__(self, base_dir, config, client):
        self.base_dir = base_dir
        self.config = config
        self.client = client

        # Load validation dataset
        self.validation_data = []
        path_to_val_file = (
            Path(self.base_dir)
            / "financial_ner/data/dataset"
            / self.config["validation_file"]
        )

        with open(path_to_val_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.validation_data.append(json.loads(line))

        # Sample validation data
        sample_size = int(
            len(self.validation_data) * self.config["validation_sample_ratio"]
        )
        self.validation_data = random.sample(self.validation_data, sample_size)

        # Generation statistics for early stopping of individuals
        self.prev_gen_avg = None
        self.prev_gen_std = None
        self.early_stopped_count = 0

    def update_generation_stats(self, avg, std):
        """
        Update statistics from the previous generation.

        Args:
            avg (float): average fitness of previous generation
            std (float): std deviation of fitness of previous generation
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

        Returns:
            float: average f1 score across validation set
        """
        # Shuffle validation data for this individual
        shuffled_validation_data = random.sample(self.validation_data, len(self.validation_data))
        
        # Calculate early stopping checkpoint based on fraction of val set
        early_stop_fraction = self.config["early_stop_checkpoint_fraction"]
        early_stop_checkpoint = int(len(shuffled_validation_data) * early_stop_fraction)

        std_multiplier = self.config["early_stop_std_multiplier"]

        scores = []

        for idx, example in enumerate(shuffled_validation_data):
            text = example["text"]
            parts = text.split("Assistant Prediction:")
            user_part = parts[0][6:].strip()  # Also removes "User: " prefix
            prediction_part = parts[1].strip()

            response = get_llm_response(
                config=self.config,
                client=self.client,
                individual=individual,
                input_text=user_part,
                response_schema=XBRLResponse,
            )

            # Convert prediction_part to comparable dict
            if prediction_part == "No XBRL associated data.":
                gold_labels = {}
            else:
                gold_labels = json.loads(prediction_part.replace("'", '"'))

            if response is None:
                logging.info(f"Response is None, using score 0.0, but something is probably wrong.")
                scores.append(0.0)
            else:
                payload = response.root if hasattr(response, "root") else response  # .root to access the actual data
                score = self.compare_json_objects(gold_labels, payload)
                scores.append(score)

            # Check for early stopping after reaching the checkpoint
            if (self.prev_gen_avg is not None and
                    idx + 1 == early_stop_checkpoint):

                current_avg = sum(scores) / len(scores)
                threshold = self.prev_gen_avg - (std_multiplier * self.prev_gen_std)

                if current_avg < threshold:
                    # Individual is performing poorly, but only stop if probability
                    # to maintain exploration
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
        Compares two JSON objects (dictionaries) and calculates the f1 score.

        Args:
            gold: Dict containing the gold labels.
            pred: Dict containing the predicted labels.
        Returns:
            float: f1 score
        """

        if not gold and not pred:
            return 1.0  # perfect match if both are empty
        if not gold or not pred:
            return 0.0

        tp = 0
        fp = 0
        fn = 0

        for key, gold_values in gold.items():

            pred_values = pred.get(key, [])  # default to empty list if key not found

            for value in gold_values:
                if value in pred_values:
                    tp += 1  # correctly predicted
                else:
                    fn += 1  # in gold but not predicted

            fp += len(set(pred_values) - set(gold_values))  # predicted but not in gold

        extra_keys = set(pred.keys()) - set(gold.keys())  # Keys in pred but not in gold

        for key in extra_keys:
            # all values in these keys are incorrect
            pred_values = pred[key]
            fp += len(pred_values)

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


def create_evaluator(base_dir, eval_config, client):
    """
    Function factory because GA requires a single-argument function.
    """
    evaluator = Evaluator(base_dir, eval_config, client)
    return evaluator.evaluate_individual


def main():
    config = load_config()

    # Set up paths
    task_dir = Path(__file__).parent
    base_dir = task_dir.parent

    # Initialize wandb
    run = init_wandb(task_name=config["task"], config=config)

    # Initialize OpenAI-compatible client for evaluation
    logging.info("Creating OpenAI client for evaluation...")

    client = OpenAI(
        base_url=f'http://localhost:{os.getenv("LLM_PORT", "8000")}/v1',
        api_key=os.getenv("LLM_API_KEY", "prompt-paper"),
    )
    
    eval_fn = create_evaluator(
        base_dir=str(base_dir), eval_config=config["evaluation"], client=client
    )

    run_evolve_stage(
        task=config["task"],
        base_dir=str(base_dir),
        config=config["evolution"],
        eval_fn=eval_fn,
    )

    finish_wandb()

    logging.info("Evolution stage completed successfully!")


if __name__ == "__main__":
    start_vllm_servers()
    main()
