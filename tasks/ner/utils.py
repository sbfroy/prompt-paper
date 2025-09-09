import pandas as pd
from pathlib import Path
from conllu import parse

def create_df(file_path):

    """
    Reads a CONLL-U formatted file, extracts words and enitity labels, 
    and returns a df with the words and labels for each sentence.
    """

    # Open and read the CoNLL-U file
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = parse(f.read())

    data = {'full_text': [], 'words': [], 'labels': []}

    for sentence in sentences:

        full_text = sentence.metadata['text'] # Extract the full text
        words = [word['form'] for word in sentence] # Extract the words
        labels = [word['misc']['name'] for word in sentence] # Extract the entity labels

        data['full_text'].append(full_text)
        data['words'].append(words)
        data['labels'].append(labels)

    return pd.DataFrame(data)

def get_label_mappings():
    """
    Get label mappings for NER task.
    
    Returns:
        Tuple of (label_to_id, id_to_label) dictionaries
    """
    label_to_id = {"O": 0, "B-FELT": 1, "I-FELT": 2}
    id_to_label = {v: k for k, v in label_to_id.items()}
    return label_to_id, id_to_label
