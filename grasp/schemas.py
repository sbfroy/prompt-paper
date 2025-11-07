from typing import List

from pydantic import BaseModel, Field


class InputExample(BaseModel):
    """Schema for the examples entering the pipeline."""

    id: str
    input: str
    output: str


class InputDataset(BaseModel):
    """Schema for the input dataset."""

    examples: list[InputExample]
    task_type: str


class EmbeddedExample(BaseModel):
    """Schema after embedding stage."""

    id: str
    input: str
    output: str
    embedding: List[float] = Field(..., min_items=1)  # Ensures non-empty vectors


class EmbeddedDataset(BaseModel):
    """Schema for the embedded dataset."""

    examples: list[EmbeddedExample]
    task_type: str


class ClusterExample(BaseModel):
    """Schema for a single example within a cluster."""

    id: str
    input: str
    output: str
    membership_probability: float  # HDBSCAN probability of belonging to the cluster


class Cluster(BaseModel):
    """Schema for a cluster."""

    cluster_id: int
    examples: list[ClusterExample]


class ClusterDataset(BaseModel):
    """Schema for the cluster dataset."""

    clusters: list[Cluster]
    task_type: str
