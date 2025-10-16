import json
from pathlib import Path
from pydantic import RootModel
import logging
import random

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

from proptimize.stages.evolve import get_llm_response

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

            if not response:
                logging.info(f"Empty response for sentence, using score 0.0")
                scores.append(0.0)
            else:
                score = compare_json_objects(gold_labels, response)
                scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0


def compare_json_objects(gold, pred):
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
