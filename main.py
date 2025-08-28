# Main orchestration script

"""
Takes in data (scope or dataset) 

Either generates ICL examples or just uses existing ones


Define what form the data needs to be in for this part to recieve it and then generates this:

This can work for the NER if I transform the conllu set to a text form (which i have to do for the ICL examples i need B-FELT AND I-FELT and etc)
this means i need to add an addditonal script to the task folder that transforms the data so it can go trhough teh first step accordingly. I also need
to for the evaluation script adjust the input format based on the prev defined format.

Makes a JSONL: {"example_id":<example_id>, "text": <text>}
"""

"""
Clustering

Takes all the examples and makes embeddings

Then clusters

Forwards a JSONL: {"cluster_id": [{"example_id": <example_id>, "text": <text>}, ...], ...}
"""


"""
GA

This one can easily use the clustering output
"""