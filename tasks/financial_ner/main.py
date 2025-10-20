import sys
import logging
from pathlib import Path
from openai import OpenAI
import yaml
import os
import json
from pydantic import RootModel
import logging
import random

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file

logging.basicConfig(level=logging.INFO)

# suppress the batch logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))  # step out to 'prompt-paper'

from proptimize.wandb_utils import init_wandb, finish_wandb
from proptimize.stages.cluster import run_cluster_stage
from proptimize.stages.evolve import run_evolve_stage
from proptimize.stages.evolve import get_llm_response
from proptimize.run_vllm import start_vllm_servers


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

    def evaluate_individual(self, individual):
        """
        Evaluate an individual on the validation set.

        Returns:
            float: average f1 score across validation set
        """

        scores = []

        for example in self.validation_data:
            text: str = example["text"]
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

            if not response:
                logging.info(f"Empty response for sentence, using score 0.0")
                scores.append(0.0)
            else:
                score = compare_json_objects(gold_labels, response)
                scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0


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


def load_config():  # Loading the config file
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_evaluator(base_dir, eval_config, client):
    """
    Function factory because GA requires a single-argument function.
    """
    evaluator = Evaluator(base_dir, eval_config, client)
    return evaluator.evaluate_individual


def run_pipeline():
    config = load_config()

    # Set up paths
    task_dir = Path(__file__).parent
    base_dir = task_dir.parent

    # Initialize DataManager
    # data_manager = DataManager(config["task"], str(base_dir))

    # Initialize wandb
    run = init_wandb(task_name=config["task"], config=config)

    # ====== Run CLUSTERING STAGE ======

    run_cluster_stage(
        task=config["task"], base_dir=str(base_dir), config_dict=config["clustering"]
    )

    # Load cluster dataset from wandb
    # cluster_dataset = data_manager.load_cluster_dataset()

    # ====== RUN EVOLUTION STAGE ======

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

    logging.info(f"Pipeline completed successfully!")


if __name__ == "__main__":
    start_vllm_servers()
    run_pipeline()
