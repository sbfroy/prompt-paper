import spacy
import re

"""
Data processing, cleaning, and structuring for LLM consumption
"""

_spacy_model = None

def _get_spacy_model(model_name):
    global _spacy_model
    if _spacy_model is None:
        _spacy_model = spacy.load(model_name)
    return _spacy_model

def split_into_sentences(text, model_name):
    nlp = _get_spacy_model(model_name)
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def clean_text(text):
    # Should probably do some minor cleaning
    pass
