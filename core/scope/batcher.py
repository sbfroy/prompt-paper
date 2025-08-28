import tiktoken
from .processors import split_into_sentences

enc = tiktoken.get_encoding("cl100k_base")

def create_batches(text, target_token_len, spacy_model):
    """
    Splits text into batches with some overlap to maintain context.
    TODO: In the future should be a more semantic batcher (paragraphs, section breaks, etc.)
    """
    sentences = split_into_sentences(text, spacy_model)

    batches = []
    current_batch = ""
    batch_id = 0
    
    for sentence in sentences:
        sentence_len = len(enc.encode(sentence))
        batch_len = len(enc.encode(current_batch))
    
        # Check if there is room for the next sentence
        if batch_len + sentence_len > target_token_len and current_batch:
            batch_text = current_batch.strip()
            batches.append({
                'id': batch_id,
                'text': batch_text
            })

            current_batch = sentence
            batch_id += 1
        else:
            current_batch += " " + sentence if current_batch else sentence
    
    # Add final batch
    if current_batch.strip():
        batch_text = current_batch.strip()
        batches.append({
            'id': batch_id,
            'text': batch_text
        })
    
    return batches
