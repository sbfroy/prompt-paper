import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

def parse_conllu_file(file_path: Path) -> List[Dict]:
    """
    Parse a CONLLU file and return a list of sentence dictionaries.
    
    Args:
        file_path: Path to the CONLLU file
        
    Returns:
        List of dictionaries with keys: 'sent_id', 'full_text', 'words', 'labels'
    """
    sentences = []
    current_sentence = {
        'sent_id': None,
        'full_text': None,
        'words': [],
        'labels': []
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                # Empty line indicates end of sentence
                if current_sentence['sent_id'] is not None:
                    sentences.append(current_sentence)
                    current_sentence = {
                        'sent_id': None,
                        'full_text': None,
                        'words': [],
                        'labels': []
                    }
                continue
                
            if line.startswith('# sent_id'):
                current_sentence['sent_id'] = line.split('=')[1].strip()
            elif line.startswith('# text'):
                current_sentence['full_text'] = line.split('=', 1)[1].strip()
            elif not line.startswith('#'):
                # Parse token line
                parts = line.split('\t')
                if len(parts) >= 10:  # Standard CONLLU format has 10 columns
                    word = parts[1]
                    # Label is in the last column (MISC field)
                    misc_field = parts[9]
                    if 'name=' in misc_field:
                        label = misc_field.split('name=')[1]
                    else:
                        label = 'O'  # Default label
                    
                    current_sentence['words'].append(word)
                    current_sentence['labels'].append(label)
        
        # Don't forget the last sentence if file doesn't end with empty line
        if current_sentence['sent_id'] is not None:
            sentences.append(current_sentence)
    
    return sentences

def create_df(file_path: Path) -> pd.DataFrame:
    """
    Create a pandas DataFrame from a CONLLU file.
    
    Args:
        file_path: Path to the CONLLU file
        
    Returns:
        DataFrame with columns: 'sent_id', 'full_text', 'words', 'labels'
    """
    sentences = parse_conllu_file(file_path)
    return pd.DataFrame(sentences)

def get_label_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Get label mappings for NER task.
    
    Returns:
        Tuple of (label_to_id, id_to_label) dictionaries
    """
    label_to_id = {"O": 0, "B-FELT": 1, "I-FELT": 2}
    id_to_label = {v: k for k, v in label_to_id.items()}
    return label_to_id, id_to_label
