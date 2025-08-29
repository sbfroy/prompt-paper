from pydantic import BaseModel, Field, conlist
from enum import Enum

class TaskType(str, Enum):
    NER = "ner"

class InputExample(BaseModel):
    """Schema for the examples entering the pipeline"""
    example_id: str
    text: str

class InputDataset(BaseModel):
    """Schema for the input dataset"""
    examples: list[InputExample]
    task_type: TaskType

class EmbeddedExample(BaseModel):
    """Schema after embedding stage"""
    example_id: str
    text: str
    embedding: conlist(float, min_items=1) # Ensures non-empty vectors

class EmbeddedDataset(BaseModel):
    """Schema for the embedded dataset"""
    examples: list[EmbeddedExample]
    task_type: TaskType

class ClusterExample(BaseModel):
    """Schema for a single example within a cluster"""
    example_id: str
    text: str
    membership_probability: float # HDBSCAN probability of belonging to the cluster

class Cluster(BaseModel):
    """Schema for a cluster"""
    cluster_id: int
    examples: list[ClusterExample]

class ClusterDataset(BaseModel):
    """Schema for the cluster dataset"""
    clusters: list[Cluster]
    task_type: TaskType