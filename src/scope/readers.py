from .loaders import read_pdf

"""
Data processing, structuring, maybe batching, and etc... just general prep for LLM consumption
"""

def from_pdf(path):
    return read_pdf(path)