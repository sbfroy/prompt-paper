import torch
import logging
from torchmetrics import Precision, Recall, F1Score
from sklearn.metrics import classification_report
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)

from .utils import create_df, get_label_mappings
from core.stages.evolve import get_llm_response
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class Evaluator:    
    def __init__(self, base_dir, config, llm_instance, sampling_params):
        self.base_dir = base_dir
        self.config = config
        self.llm_instance = llm_instance
        self.sampling_params = sampling_params
    
    def evaluate_individual(self, individual):
        """
        Evaluate an individual on the NER validation dataset.
        
        """
        # Load validation data
        val_df = create_df(Path(self.base_dir) / 'ner/data/dataset' / self.config["validation_file"])
        # val_df_sample = val_df.iloc[:int(len(val_df) * config["validation_sample_ratio"])]
        # Uses consistent random sampling instead
        val_df_sample = val_df.sample(n=int(len(val_df) * self.config["validation_sample_ratio"]), random_state=42)
        
        scores = []
        for _, row in val_df_sample.iterrows():
            sentence = row['full_text']
            tokens = row['words']
            true_labels = row['labels']
            
            response = get_llm_response(
                prompt_template=self.config["prompt_template"],
                individual=individual,
                input_text=sentence,
                llm_instance=self.llm_instance,
                sampling_params=self.sampling_params,
            )
            
            # Handle empty responses (from client errors)
            if not response:
                logging.info(f"Empty response for sentence, using score 0.0")
                scores.append(0.0)
            else:
                score = evaluate_llm_response(response, true_labels, tokens)
                scores.append(score)
        
        if not scores:
            return 0.0
        
        avg_score = sum(scores) / len(scores)
        return avg_score
        
def evaluate_llm_response(response_text, true_labels, tokens):
    """
    Evaluates LLM response against true labels and returns F1 score.
    
    Args:
        response_text: Raw text response from LLM
        true_labels: List of true labels for the tokens
        tokens: List of tokens in the test sentence
    
    Returns:
        float: F1 score
    """
    # Get label mappings
    label_to_id, id_to_label = get_label_mappings()
    
    # Parse LLM response
    entities = defaultdict(list)
    
    for line in response_text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            word, label = parts[0], parts[1]
            entities[word].append(label)
    
    # Convert to predicted labels
    pred_labels = []
    word_counts = defaultdict(int)  # Track occurrences of each word

    for token in tokens:
        if token in entities and word_counts[token] < len(entities[token]):
            pred_labels.append(entities[token][word_counts[token]])  # Get the label in order
            word_counts[token] += 1  # Increment occurrence counter
        else:
            pred_labels.append("O")  # Default to "O" if missing

    # Convert to IDs
    pred_ids = []
    for label in pred_labels:
        if label in label_to_id:
            pred_ids.append(label_to_id[label])
        else:
            pred_ids.append(label_to_id["O"])  # Default to "O" 

    true_ids = [label_to_id.get(label, label_to_id["O"]) for label in true_labels]
    
    metrics = evaluate(true_ids, pred_ids, id_to_label)
    
    return metrics['f1']

def evaluate(preds, labels, id_to_label):
    """
    Calculate evaluation metrics for predictions vs labels.
    
    Args:
        preds: List of predicted label IDs
        labels: List of true label IDs  
        id_to_label: Mapping from ID to label string
        
    Returns:
        dict: Dictionary with precision, recall, f1, and classification report
    """
    num_classes = len(id_to_label)
    precision_metric = Precision(task='multiclass', num_classes=num_classes, average='macro')
    recall_metric = Recall(task='multiclass', num_classes=num_classes, average='macro')
    f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='macro')

    preds_tensor = torch.tensor(preds, dtype=torch.int64)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    precision_score = precision_metric(preds_tensor, labels_tensor).item()
    recall_score = recall_metric(preds_tensor, labels_tensor).item()
    f1_score = f1_metric(preds_tensor, labels_tensor).item()

    # Classification report
    pred_labels = [id_to_label[pred] for pred in preds]
    true_labels = [id_to_label[label] for label in labels]
    report = classification_report(true_labels, pred_labels, zero_division=0, output_dict=True)

    return {
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'classification_report': report
    }
