"""
Evaluate Best Individuals from GA Evolution

Systematically evaluates:
  A. 6 best individuals (3 dataset sizes × 2 banks)
  B. 6 random baselines (2 banks × 3 seeds)
  C. 1 zero-shot baseline (no ICL examples)

Results are saved as a WandB artifact (JSON + CSV).
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import RootModel

from grasp.wandb_utils import init_wandb, save_artifact, finish_wandb
from grasp.stages.client import get_llm_response
from grasp.run_vllm import start_vllm_servers
from grasp.data_manager import DataManager
from grasp.schemas import ClusterDataset

load_dotenv()

logging.basicConfig(level=logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# CONFIGURATION: Best Individuals
# =============================================================================
# Each entry is a list of (cluster_id, example_id) tuples.
# Extract these from your checkpoint.json files:
#   checkpoint["hall_of_fame"][0]["genes"]
#   → [{"cluster_id": 3, "example_id": "1042"}, ...]
#   → [(3, "1042"), ...]
#
# "synthetic" individuals must index into the SYNTHETIC cluster dataset.
# "nonsynthetic" individuals must index into the NON-SYNTHETIC (real) cluster dataset.
# =============================================================================

BEST_INDIVIDUALS = {
    # Synthetic example bank (use_real=False)
    "synthetic_k5000": [
        # (cluster_id, example_id),
        # (cluster_id, example_id),
        # (cluster_id, example_id),
        # (cluster_id, example_id),
        # (cluster_id, example_id),
    ],
    "synthetic_k500": [
        # TODO: fill in from checkpoint
    ],
    "synthetic_k50": [
        # TODO: fill in from checkpoint
    ],

    # Non-synthetic (real) example bank (use_real=True)
    "nonsynthetic_k5000": [
        # TODO: fill in from checkpoint
    ],
    "nonsynthetic_k500": [
        # TODO: fill in from checkpoint
    ],
    "nonsynthetic_k50": [
        # TODO: fill in from checkpoint
    ],
}

# Random baseline seeds (deterministic for reproducibility)
RANDOM_SEEDS = [42, 123, 456]

# Number of ICL examples per individual (must match evolution subset_size)
SUBSET_SIZE = 5


# =============================================================================
# Evaluation logic (reuses existing Evaluator from run_evolve.py)
# =============================================================================

class XBRLResponse(RootModel[dict[str, list[str]]]):
    """Pydantic model for XBRL entity extraction responses."""
    pass


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_example_index(cluster_dataset: ClusterDataset) -> dict:
    """Build lookup: (cluster_id, example_id) → (cluster_id, ClusterExample)."""
    index = {}
    for cluster in cluster_dataset.clusters:
        for example in cluster.examples:
            index[(cluster.cluster_id, example.id)] = (cluster.cluster_id, example)
    return index


def reconstruct_individual(gene_tuples, example_index, condition_name):
    """
    Reconstruct an individual from (cluster_id, example_id) tuples.

    Args:
        gene_tuples: List of (cluster_id, example_id) pairs.
        example_index: Lookup dict from build_example_index.
        condition_name: Name of the condition (for error messages).

    Returns:
        List of (cluster_id, ClusterExample) tuples (same format as GA individual).
    """
    individual = []
    for cluster_id, example_id in gene_tuples:
        key = (cluster_id, str(example_id))
        if key not in example_index:
            raise ValueError(
                f"[{condition_name}] Example (cluster_id={cluster_id}, "
                f"example_id={example_id}) not found in cluster dataset. "
                f"Are you using the correct bank (synthetic vs non-synthetic)?"
            )
        individual.append(example_index[key])
    return individual


def create_random_individual(cluster_dataset: ClusterDataset, rng: random.Random):
    """
    Create a random individual by sampling from the cluster dataset.
    Mirrors GA._create_individual() but uses a local RNG for isolation.
    """
    individual = []
    for _ in range(SUBSET_SIZE):
        cluster = rng.choice(cluster_dataset.clusters)
        example = rng.choice(cluster.examples)
        individual.append((cluster.cluster_id, example))
    return individual


def evaluate_on_test_set(individual, test_data, config, client):
    """
    Evaluate an individual on the test set. No early stopping.
    Reuses the same logic as Evaluator.evaluate_on_test_set() from run_evolve.py.

    Args:
        individual: List of (cluster_id, ClusterExample) tuples, or None for zero-shot.
        test_data: List of {"input": str, "output": str} dicts.
        config: Evaluation config dict.
        client: OpenAI-compatible client.

    Returns:
        Dict with micro_precision, micro_recall, micro_f1, macro_f1.
    """
    all_label_stats = {}

    for example in test_data:
        user_input = example["input"]
        gold_output = example["output"]

        # Format ICL examples (empty for zero-shot)
        if individual is not None and len(individual) > 0:
            formatted_examples = []
            for _, icl_example in individual:
                formatted_example = (
                    f"Input: {icl_example.input}\n"
                    f"Output: {icl_example.output}"
                )
                formatted_examples.append(formatted_example)
            examples_text = "\n\n".join(formatted_examples)
        else:
            examples_text = "(No examples provided)"

        # Construct prompt
        user_prompt = config["user_prompt"].format(
            examples=examples_text,
            input_text=user_input,
        )

        messages = [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ]

        response = get_llm_response(
            client=client,
            messages=messages,
            response_schema=XBRLResponse,
            temperature=config["temperature"],
            guided_decoding_backend=config["guided_decoding_backend"],
        )

        # Parse gold labels
        if gold_output == "No XBRL associated data.":
            gold_labels = {}
        else:
            gold_labels = json.loads(gold_output.replace("'", '"'))

        if response is None:
            logging.warning("Response is None, treating as empty prediction.")
            response = {}

        # Compute per-label TP/FP/FN and aggregate
        example_stats = compute_example_stats(gold_labels, response)
        for label, stats in example_stats.items():
            if label not in all_label_stats:
                all_label_stats[label] = {"tp": 0, "fp": 0, "fn": 0}
            all_label_stats[label]["tp"] += stats["tp"]
            all_label_stats[label]["fp"] += stats["fp"]
            all_label_stats[label]["fn"] += stats["fn"]

    if not all_label_stats:
        return {
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
        }

    return compute_metrics_from_stats(all_label_stats)


def compute_example_stats(gold, pred):
    """Compute per-label TP/FP/FN for a single example. (Same as Evaluator.compute_example_stats)"""
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


def compute_metrics_from_stats(all_label_stats):
    """Compute micro/macro metrics from aggregated stats. (Same as Evaluator.compute_metrics_from_stats)"""
    total_tp = sum(s["tp"] for s in all_label_stats.values())
    total_fp = sum(s["fp"] for s in all_label_stats.values())
    total_fn = sum(s["fn"] for s in all_label_stats.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    label_f1s = []
    for stats in all_label_stats.values():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        label_f1s.append(f1)

    macro_f1 = sum(label_f1s) / len(label_f1s) if label_f1s else 0.0

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


# =============================================================================
# Main evaluation driver
# =============================================================================

def run_all_evaluations(config, client, data_manager):
    """
    Run all 13 evaluation conditions and return consolidated results.
    """
    eval_config = {
        **config["evaluation"],
        "use_real": config["dataset"]["use_real"],
    }

    # ------------------------------------------------------------------
    # Load test data (always real)
    # ------------------------------------------------------------------
    logging.info("Loading test dataset...")
    test_dataset = data_manager.load_input_dataset("test", use_real=True)
    test_data = [
        {"input": ex.input, "output": ex.output}
        for ex in test_dataset.examples
    ]
    logging.info(f"Test set size: {len(test_data)} examples")

    # ------------------------------------------------------------------
    # Load both cluster datasets (for example lookup & random sampling)
    # ------------------------------------------------------------------
    logging.info("Loading synthetic cluster dataset...")
    synthetic_clusters = data_manager.load_cluster_dataset(use_real=False)
    synthetic_clusters.clusters = [
        c for c in synthetic_clusters.clusters if c.cluster_id != -1
    ]
    synthetic_index = build_example_index(synthetic_clusters)
    n_synthetic = sum(len(c.examples) for c in synthetic_clusters.clusters)
    logging.info(
        f"Synthetic bank: {len(synthetic_clusters.clusters)} clusters, "
        f"{n_synthetic} examples"
    )

    logging.info("Loading non-synthetic (real) cluster dataset...")
    nonsynthetic_clusters = data_manager.load_cluster_dataset(use_real=True)
    nonsynthetic_clusters.clusters = [
        c for c in nonsynthetic_clusters.clusters if c.cluster_id != -1
    ]
    nonsynthetic_index = build_example_index(nonsynthetic_clusters)
    n_nonsynthetic = sum(len(c.examples) for c in nonsynthetic_clusters.clusters)
    logging.info(
        f"Non-synthetic bank: {len(nonsynthetic_clusters.clusters)} clusters, "
        f"{n_nonsynthetic} examples"
    )

    results = []

    def run_single_eval(condition_name, individual, bank_name, example_indices_desc):
        """Run one evaluation and record results."""
        logging.info(f"\n{'='*60}")
        logging.info(f"Running: {condition_name}")
        logging.info(f"  Bank: {bank_name}")
        logging.info(f"  Examples: {example_indices_desc}")
        logging.info(f"{'='*60}")

        metrics = evaluate_on_test_set(individual, test_data, eval_config, client)

        logging.info(
            f"  → micro_f1={metrics['micro_f1']:.4f}  "
            f"micro_p={metrics['micro_precision']:.4f}  "
            f"micro_r={metrics['micro_recall']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

        results.append({
            "condition": condition_name,
            "bank": bank_name,
            "example_indices": example_indices_desc,
            "micro_precision": metrics["micro_precision"],
            "micro_recall": metrics["micro_recall"],
            "micro_f1": metrics["micro_f1"],
            "macro_f1": metrics["macro_f1"],
        })

        return metrics

    # ------------------------------------------------------------------
    # A. Best Individuals (6 runs)
    # ------------------------------------------------------------------
    logging.info("\n\n" + "#" * 60)
    logging.info("# SECTION A: Best Individuals (6 runs)")
    logging.info("#" * 60)

    for condition_name, gene_tuples in BEST_INDIVIDUALS.items():
        if not gene_tuples:
            logging.warning(
                f"Skipping {condition_name}: no genes specified (TODO placeholder)."
            )
            continue

        # Determine which bank to use
        is_synthetic = condition_name.startswith("synthetic")
        bank_name = "synthetic" if is_synthetic else "nonsynthetic"
        example_index = synthetic_index if is_synthetic else nonsynthetic_index

        individual = reconstruct_individual(gene_tuples, example_index, condition_name)

        # Log which examples we resolved
        resolved_desc = json.dumps(
            [(cid, ex.id) for cid, ex in individual], indent=None
        )
        logging.info(f"[{condition_name}] Resolved {len(individual)} examples from {bank_name} bank")

        run_single_eval(condition_name, individual, bank_name, resolved_desc)

    # ------------------------------------------------------------------
    # B. Random Baselines (6 runs: 2 banks × 3 seeds)
    # ------------------------------------------------------------------
    logging.info("\n\n" + "#" * 60)
    logging.info("# SECTION B: Random Baselines (6 runs)")
    logging.info("#" * 60)

    for bank_name, cluster_dataset in [
        ("synthetic", synthetic_clusters),
        ("nonsynthetic", nonsynthetic_clusters),
    ]:
        for seed_idx, seed in enumerate(RANDOM_SEEDS):
            condition_name = f"random_{bank_name}_seed{seed}"
            rng = random.Random(seed)

            individual = create_random_individual(cluster_dataset, rng)

            indices_desc = json.dumps(
                [(cid, ex.id) for cid, ex in individual], indent=None
            )
            logging.info(
                f"[{condition_name}] Sampled {len(individual)} random examples "
                f"from {bank_name} bank (seed={seed})"
            )

            run_single_eval(condition_name, individual, bank_name, indices_desc)

    # Compute random baseline summary stats
    for bank_name in ["synthetic", "nonsynthetic"]:
        bank_random = [
            r for r in results
            if r["condition"].startswith(f"random_{bank_name}")
        ]
        if bank_random:
            for metric in ["micro_f1", "macro_f1", "micro_precision", "micro_recall"]:
                values = [r[metric] for r in bank_random]
                mean_val = sum(values) / len(values)
                std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
                logging.info(
                    f"Random {bank_name} {metric}: "
                    f"mean={mean_val:.4f} std={std_val:.4f}"
                )

            # Add summary row
            for metric in ["micro_f1", "macro_f1", "micro_precision", "micro_recall"]:
                values = [r[metric] for r in bank_random]
                mean_val = sum(values) / len(values)
                std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
                # Store on first random result for this bank
                bank_random[0][f"{metric}_mean"] = mean_val
                bank_random[0][f"{metric}_std"] = std_val

    # ------------------------------------------------------------------
    # C. Zero-Shot Baseline (1 run)
    # ------------------------------------------------------------------
    logging.info("\n\n" + "#" * 60)
    logging.info("# SECTION C: Zero-Shot Baseline (1 run)")
    logging.info("#" * 60)

    run_single_eval("zero_shot", None, "none", "none")

    return results


def save_results_artifact(results, config):
    """Save results as a WandB artifact (JSON)."""
    # Build summary with random baseline aggregations
    random_summaries = {}
    for bank_name in ["synthetic", "nonsynthetic"]:
        bank_random = [
            r for r in results
            if r["condition"].startswith(f"random_{bank_name}")
        ]
        if bank_random:
            summary = {}
            for metric in ["micro_f1", "macro_f1", "micro_precision", "micro_recall"]:
                values = [r[metric] for r in bank_random]
                mean_val = sum(values) / len(values)
                std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
                summary[f"{metric}_mean"] = round(mean_val, 6)
                summary[f"{metric}_std"] = round(std_val, 6)
            summary["individual_runs"] = [
                {
                    "condition": r["condition"],
                    "micro_f1": round(r["micro_f1"], 6),
                    "macro_f1": round(r["macro_f1"], 6),
                    "micro_precision": round(r["micro_precision"], 6),
                    "micro_recall": round(r["micro_recall"], 6),
                }
                for r in bank_random
            ]
            random_summaries[f"random_{bank_name}"] = summary

    artifact_data = {
        "description": "Evaluation of best GA individuals, random baselines, and zero-shot baseline on test set",
        "subset_size": SUBSET_SIZE,
        "random_seeds": RANDOM_SEEDS,
        "all_results": [
            {k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}
            for r in results
        ],
        "random_baseline_summaries": random_summaries,
    }

    artifact = save_artifact(
        data=artifact_data,
        artifact_name=f"{config['task']}_best_individuals_evaluation",
        artifact_type="evaluation_results",
    )
    logging.info(f"Results saved as WandB artifact: {artifact.name}")

    return artifact


def print_results_table(results):
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 100)
    print(
        f"{'Condition':<35} {'Bank':<15} "
        f"{'micro_f1':>10} {'micro_p':>10} {'micro_r':>10} {'macro_f1':>10}"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r['condition']:<35} {r['bank']:<15} "
            f"{r['micro_f1']:>10.4f} {r['micro_precision']:>10.4f} "
            f"{r['micro_recall']:>10.4f} {r['macro_f1']:>10.4f}"
        )

    # Print random baseline summaries
    print("-" * 100)
    for bank_name in ["synthetic", "nonsynthetic"]:
        bank_random = [
            r for r in results
            if r["condition"].startswith(f"random_{bank_name}")
        ]
        if bank_random:
            for metric in ["micro_f1", "macro_f1"]:
                values = [r[metric] for r in bank_random]
                mean_val = sum(values) / len(values)
                std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
                print(
                    f"  random_{bank_name} {metric}: "
                    f"mean={mean_val:.4f}  std={std_val:.4f}"
                )

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate best GA individuals, random baselines, and zero-shot"
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Skip starting vLLM server (assume it's already running)",
    )
    args = parser.parse_args()

    config = load_config()
    task_dir = Path(__file__).parent
    base_dir = task_dir.parent

    # Initialize WandB
    init_wandb(task_name=config["task"], config={
        "evaluation_type": "best_individuals_comparison",
        "subset_size": SUBSET_SIZE,
        "random_seeds": RANDOM_SEEDS,
    })

    # Initialize OpenAI client
    client_timeout = config.get("evaluation", {}).get("client_timeout", 120.0)
    client = OpenAI(
        base_url=f'http://localhost:{os.getenv("LLM_PORT", "8000")}/v1',
        api_key=os.getenv("LLM_API_KEY", "prompt-paper"),
        timeout=client_timeout,
    )

    data_manager = DataManager(config["task"], str(base_dir))

    # Run all evaluations
    results = run_all_evaluations(config, client, data_manager)

    # Print summary
    print_results_table(results)

    # Save as WandB artifact
    save_results_artifact(results, config)

    finish_wandb()
    logging.info("All evaluations completed!")


if __name__ == "__main__":
    # Quick parse to check --no-vllm before starting servers
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--no-vllm", action="store_true")
    _pre_args, _ = _pre.parse_known_args()

    if not _pre_args.no_vllm:
        start_vllm_servers(start_embedding=False, start_LLM=True)

    main()
