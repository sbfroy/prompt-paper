"""
Financial NER - Test Multiple Prompt Configurations

This script tests 5 different prompt configurations on the same test set:
- not_syn_500: IDs from financial_ner_v2_cluster_dataset_real
- not_syn_2500: IDs from financial_ner_v2_cluster_dataset_real
- syn_5000: IDs from financial_ner_v2_cluster_dataset
- syn_500: IDs from financial_ner_v2_cluster_dataset
- syn_50: IDs from financial_ner_v2_cluster_dataset

All configurations are tested on financial_ner_v2_input_dataset_real_test
Results are stored as a single artifact with F1, precision, and recall metrics.
"""

import sys
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import RootModel

from grasp.wandb_utils import init_wandb, finish_wandb, save_artifact
from grasp.stages.client import get_llm_response
from grasp.run_vllm import start_vllm_servers
from grasp.data_manager import DataManager
from grasp.schemas import ClusterExample

load_dotenv()

logging.basicConfig(level=logging.INFO)

# Suppress verbose logging from external libraries
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class XBRLResponse(RootModel[dict[str, list[str]]]):
    """Pydantic model for XBRL entity extraction responses."""
    pass


@dataclass
class ConfigurationResult:
    """Results for a single configuration."""
    config_name: str
    example_ids: List[str]
    artifact_name: str
    f1_score: float
    precision: float
    recall: float
    num_test_examples: int


@dataclass
class TestResults:
    """Complete test results for all configurations."""
    configurations: List[ConfigurationResult]
    test_artifact: str
    timestamp: str


class ConfigurationTester:
    """Tests prompt configurations on the financial NER test set."""

    def __init__(self, data_manager, config, client):
        """
        Initialize the tester.

        Args:
            data_manager: DataManager instance for loading data
            config: Configuration dictionary
            client: OpenAI-compatible client for LLM calls
        """
        self.data_manager = data_manager
        self.config = config
        self.client = client

        # Load test dataset from wandb artifact
        test_dataset = self.data_manager.load_input_dataset(
            "test",
            use_real=True
        )
        
        # Convert to dictionary format for compatibility
        self.test_data = [
            {"input": example.input, "output": example.output}
            for example in test_dataset.examples
        ]
        
        logging.info(f"Loaded {len(self.test_data)} test examples")

    def load_examples_by_ids(
        self, 
        example_ids: List[str], 
        cluster_dataset
    ) -> List[ClusterExample]:
        """
        Load specific examples by their IDs from a cluster dataset.

        Args:
            example_ids: List of example IDs to load
            cluster_dataset: ClusterDataset object

        Returns:
            List of ClusterExample objects matching the IDs
        """
        # Build a mapping from example_id to example
        id_to_example = {}
        for cluster in cluster_dataset.clusters:
            for example in cluster.examples:
                id_to_example[example.id] = example
        
        # Retrieve examples in the order of the provided IDs
        examples = []
        for example_id in example_ids:
            str_id = str(example_id)
            if str_id in id_to_example:
                examples.append(id_to_example[str_id])
            else:
                logging.warning(f"Example ID {example_id} not found in cluster dataset")
        
        if len(examples) != len(example_ids):
            logging.warning(
                f"Only found {len(examples)} out of {len(example_ids)} requested examples"
            )
        
        return examples

    def evaluate_configuration(
        self, 
        config_name: str,
        examples: List[ClusterExample]
    ) -> Tuple[float, float, float]:
        """
        Evaluate a configuration on the test set.
        
        Args:
            config_name: Name of the configuration
            examples: List of ICL examples to use

        Returns:
            Tuple of (f1_score, precision, recall)
        """
        logging.info(f"\nEvaluating configuration: {config_name}")
        logging.info(f"Using {len(examples)} ICL examples")
        
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for idx, test_example in enumerate(self.test_data):
            if (idx + 1) % 100 == 0:
                logging.info(f"  Processed {idx + 1}/{len(self.test_data)} test examples")
            
            user_input = test_example["input"]
            gold_output = test_example["output"]

            # Format ICL examples
            formatted_examples = []
            for icl_example in examples:
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

            # Log prompt length for first test example to verify prompts differ
            if idx == 0:
                logging.info(f"  First test example prompt length: {len(user_prompt)} chars")
                logging.info(f"  ICL examples text length: {len(examples_text)} chars")

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
                    f"Response is None for test example {idx}, treating as score 0.0"
                )
                f1_scores.append(0.0)
                precision_scores.append(0.0)
                recall_scores.append(0.0)
            else:
                pred_labels = response
                # Calculate F1, precision, and recall for this example
                f1, precision, recall = self._compare_json_objects(gold_labels, pred_labels)
                f1_scores.append(f1)
                precision_scores.append(precision)
                recall_scores.append(recall)

        # Calculate average metrics across all examples
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        
        logging.info(f"Configuration {config_name} results:")
        logging.info(f"  Average Precision: {avg_precision:.4f}")
        logging.info(f"  Average Recall: {avg_recall:.4f}")
        logging.info(f"  Average F1 Score: {avg_f1:.4f}")

        return avg_f1, avg_precision, avg_recall

    @staticmethod
    def _compare_json_objects(
        gold: dict[str, list[str]], 
        pred: dict[str, list[str]]
    ) -> Tuple[float, float, float]:
        """
        Compare two JSON objects and calculate F1, precision, and recall.
        This matches the evaluation method used in run_evolve.py.

        Args:
            gold: Dictionary containing the gold labels
            pred: Dictionary containing the predicted labels

        Returns:
            Tuple of (f1_score, precision, recall)
        """
        if not gold and not pred:
            return 1.0, 1.0, 1.0  # Perfect match if both are empty

        if not gold or not pred:
            return 0.0, 0.0, 0.0

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

        return f1, precision, recall


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Run tests for all prompt configurations."""
    config = load_config()

    # Set up paths
    task_dir = Path(__file__).parent
    base_dir = task_dir.parent

    # Initialize wandb for experiment tracking
    init_wandb(task_name=config['task'], config=config)

    # Initialize OpenAI-compatible client for evaluation
    logging.info("Creating OpenAI client for evaluation...")

    client = OpenAI(
        base_url=f'http://localhost:{os.getenv("LLM_PORT", "8000")}/v1',
        api_key=os.getenv("LLM_API_KEY", "prompt-paper"),
    )

    data_manager = DataManager(config["task"], str(base_dir))

    # Create tester
    tester = ConfigurationTester(
        data_manager=data_manager,
        config={
            **config["evaluation"],
            "use_real": True
        },
        client=client
    )

    # Define configurations to test
    configurations = [
        {
            "name": "not_syn_500",
            "example_ids": ["570797", "726984", "487547", "431724", "832572"],
            "use_real": True
        },
        {
            "name": "not_syn_2500",
            "example_ids": ["83306", "767038", "573233", "230", "442564"],
            "use_real": True
        },
        {
            "name": "syn_5000",
            "example_ids": ["1505", "2074", "8801", "3370", "4143"],
            "use_real": False
        },
        {
            "name": "syn_500",
            "example_ids": ["3773", "599", "1206", "882", "5518"],
            "use_real": False
        },
        {
            "name": "syn_50",
            "example_ids": ["3754", "2952", "1940", "768", "133"],
            "use_real": False
        }
    ]

    # Run tests for all configurations
    results = []
    
    for conf in configurations:
        logging.info(f"\n{'='*80}")
        logging.info(f"Testing configuration: {conf['name']}")
        logging.info(f"{'='*80}")
        
        # Load cluster dataset for this configuration
        cluster_dataset = data_manager.load_cluster_dataset(
            use_real=conf["use_real"]
        )
        
        # Load examples for this configuration
        examples = tester.load_examples_by_ids(
            conf["example_ids"],
            cluster_dataset
        )
        
        # Evaluate configuration
        f1, precision, recall = tester.evaluate_configuration(
            conf["name"],
            examples
        )
        
        # Store results
        artifact_name = f"financial_ner_v2_cluster_dataset"
        if conf["use_real"]:
            artifact_name += "_real"
        
        result = ConfigurationResult(
            config_name=conf["name"],
            example_ids=conf["example_ids"],
            artifact_name=artifact_name,
            f1_score=f1,
            precision=precision,
            recall=recall,
            num_test_examples=len(tester.test_data)
        )
        results.append(result)

    # Create final results object
    from datetime import datetime
    test_results = TestResults(
        configurations=results,
        test_artifact="financial_ner_v2_input_dataset_real_test",
        timestamp=datetime.now().isoformat()
    )

    # Save results as artifact
    results_dict = {
        "test_artifact": test_results.test_artifact,
        "timestamp": test_results.timestamp,
        "configurations": [asdict(r) for r in test_results.configurations]
    }
    
    artifact_name = "financial_ner_v2_prompt_config_test_results"
    save_artifact(results_dict, artifact_name, "test_results")
    
    logging.info(f"\n{'='*80}")
    logging.info("FINAL RESULTS SUMMARY")
    logging.info(f"{'='*80}")
    
    for result in results:
        logging.info(f"\nConfiguration: {result.config_name}")
        logging.info(f"  Artifact: {result.artifact_name}")
        logging.info(f"  Example IDs: {result.example_ids}")
        logging.info(f"  Precision: {result.precision:.4f}")
        logging.info(f"  Recall: {result.recall:.4f}")
        logging.info(f"  F1 Score: {result.f1_score:.4f}")
    
    logging.info(f"\nResults saved to artifact: {artifact_name}")

    finish_wandb()

    logging.info("\nAll configurations tested successfully!")


if __name__ == "__main__":
    start_vllm_servers(start_embedding=False, start_LLM=True)
    main()
