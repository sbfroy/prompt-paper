from pathlib import Path
from .loaders import read_pdf
from .batcher import create_batches

def get_batches(config):
    """
    Load the whole corpus and create batches for LLM processing.
    """
    all_text = ""
    
    config = config.get('scope', {})
    test_scope = config.get('test_scope_01', {})
    documents = scan_scope_folder(test_scope.get('scope_folder'), test_scope.get('supported_types'))

    # Process each document
    for doc_info in documents:
        doc_text = load_document(doc_info)
        all_text += doc_text

    target_token_len = config.get('target_token_len')
    overlap_token_len = config.get('overlap_token_len')
    spacy_model = config.get('sentence_splitter_model')

    # Create batches
    batches = create_batches(all_text, target_token_len, overlap_token_len, spacy_model)

    return {
        'batches': batches,
        'metadata': {
            # Keep some useful metadata
            'total_batches': len(batches),
        }
    }

def scan_scope_folder(data_folder, supported_types):
    """
    Scan the data folder for documents of supported types.
    """
    documents = []
    folder_path = Path(data_folder)

    # Scan for files
    for file_path in folder_path.rglob('*'):
        file_type = file_path.suffix.lower().lstrip('.')
        if file_type in supported_types:
            documents.append({
                'name': file_path.name,
                'path': str(file_path),
                'type': file_type
            })
    
    return documents

def load_document(doc_info):
    """
    Load a document based on its type.
    """
    doc_type = doc_info['type']
    
    if doc_type == 'pdf':
        return read_pdf(doc_info['path'])
    elif doc_type == 'url':
        # Placeholder for future URL loading
        pass
