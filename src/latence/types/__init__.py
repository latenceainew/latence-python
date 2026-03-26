"""Type exports for the Latence SDK.

This module re-exports all public types for convenient importing:

    from latence.types import EmbedResponse, Entity, Usage
"""

from __future__ import annotations

# Re-export all models for type annotations
from .._models import (
    # Common
    BaseResponse,
    # Ontology
    BuildGraphResponse,
    # ColBERT
    ColBERTEmbedResponse,
    # ColPali
    ColPaliEmbedResponse,
    # Compression
    CompressMessagesResponse,
    CompressResponse,
    # Jobs
    CreditsResponse,
    CustomLabel,
    # Redaction
    DetectPIIResponse,
    # Document Intelligence
    DocumentMetadata,
    # Embedding
    EmbedResponse,
    Entity,
    EntityInput,
    # Extraction
    ExtractionConfig,
    ExtractResponse,
    JobCancelResponse,
    JobListResponse,
    JobStatus,
    JobStatusResponse,
    JobSubmittedResponse,
    KnowledgeGraph,
    Message,
    OntologyConfig,
    ProcessDocumentResponse,
    RedactionConfig,
    Relation,
    Usage,
)

__all__ = [
    # Common
    "BaseResponse",
    "CustomLabel",
    "Entity",
    "KnowledgeGraph",
    "Message",
    "Relation",
    "Usage",
    # Embedding
    "EmbedResponse",
    # ColBERT
    "ColBERTEmbedResponse",
    # ColPali
    "ColPaliEmbedResponse",
    # Compression
    "CompressMessagesResponse",
    "CompressResponse",
    # Document Intelligence
    "DocumentMetadata",
    "ProcessDocumentResponse",
    # Extraction
    "ExtractionConfig",
    "ExtractResponse",
    # Ontology
    "BuildGraphResponse",
    "EntityInput",
    "OntologyConfig",
    # Redaction
    "DetectPIIResponse",
    "RedactionConfig",
    # Jobs
    "CreditsResponse",
    "JobCancelResponse",
    "JobListResponse",
    "JobStatus",
    "JobStatusResponse",
    "JobSubmittedResponse",
]
