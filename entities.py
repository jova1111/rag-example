"""Pydantic models for request and response schemas."""

from typing import Optional
from pydantic import BaseModel


class ClassificationResult(BaseModel):
    """Classification result model."""
    tags: list[str]
    confidence: float
    justification: str
    document_length: int
    chunks_processed: int
    retrieved_documents_count: int
    retrieved_documents: Optional[list] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str
    embedding_model: str
    llm_model: str


class TextClassificationRequest(BaseModel):
    """Request model for text classification."""
    text: str
    include_context: bool = False
