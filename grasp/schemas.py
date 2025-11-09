"""Pydantic schemas for data structures used throughout the pipeline."""

from typing import List

from pydantic import BaseModel, Field


class InputExample(BaseModel):
    """Single example entering the pipeline.
    
    Attributes:
        id: Unique identifier
        input: Input text/prompt
        output: Expected output/label
    """

    id: str
    input: str
    output: str


class InputDataset(BaseModel):
    """Collection of input examples for a specific task.
    
    Attributes:
        examples: List of input examples
        task_type: Task identifier (e.g., 'ner', 'financial_ner')
    """

    examples: list[InputExample]
    task_type: str


class EmbeddedExample(BaseModel):
    """Example with vector embedding added.
    
    Attributes:
        id: Unique identifier
        input: Input text/prompt
        output: Expected output/label
        embedding: Dense vector representation
    """

    id: str
    input: str
    output: str
    embedding: List[float] = Field(..., min_items=1)


class EmbeddedDataset(BaseModel):
    """Collection of embedded examples.
    
    Attributes:
        examples: List of embedded examples
        task_type: Task identifier
    """

    examples: list[EmbeddedExample]
    task_type: str


class ClusterExample(BaseModel):
    """Example within a cluster with membership probability.
    
    Attributes:
        id: Unique identifier
        input: Input text/prompt
        output: Expected output/label
        membership_probability: HDBSCAN probability of belonging to cluster
    """

    id: str
    input: str
    output: str
    membership_probability: float


class Cluster(BaseModel):
    """Group of related examples identified by clustering.
    
    Attributes:
        cluster_id: Unique cluster identifier
        examples: List of examples in this cluster
    """

    cluster_id: int
    examples: list[ClusterExample]


class ClusterDataset(BaseModel):
    """Collection of clusters for a task.
    
    Attributes:
        clusters: List of clusters
        task_type: Task identifier
    """

    clusters: list[Cluster]
    task_type: str
