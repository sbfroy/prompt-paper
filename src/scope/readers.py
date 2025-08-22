from .loaders import read_pdf

"""
Data processing, structuring, maybe batching, and etc... just general prep for LLM consumption
"""

def from_pdf(path):
    return read_pdf(path)


# TODO: Read all scope, divide into 20K batches, done


# I will divide the full scope and context into equal-sized batches, 
# where the batch size is determined by the total token length of the 
# corpus and a chosen target token size per batch*. Each batch will then 
# be given to the LLM one by one, and I will deliberately over-generate 
# examples to maximize coverage, accepting that some may be noisy or 
# hallucinated. By keeping batches separate, I can reserve certain ones 
# exclusively for the fitness function, ensuring that evaluation examples 
# do not overlap with the training pool. After generation, the examples 
# will be filtered, clustered, and optimized to retain only the most 
# diverse and high-quality ones.

# Target token size could be 20K, research shows that after this the LLM
# starts to struggle with context retention and may produce less correct outputs.