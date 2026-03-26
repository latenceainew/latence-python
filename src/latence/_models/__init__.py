"""Pydantic models for Latence API requests and responses."""

from __future__ import annotations

from .chunking import (
    ChunkData,
    ChunkItem,
    ChunkResponse,
)
from .colbert import ColBERTEmbedResponse
from .colpali import ColPaliEmbedResponse
from .common import (
    BaseResponse,
    CustomLabel,
    Entity,
    KnowledgeGraph,
    Message,
    Relation,
    Usage,
)
from .compression import CompressMessagesResponse, CompressResponse
from .dataset_intelligence_service import (
    DatasetIntelligenceDeltaSummary,
    DatasetIntelligenceResponse,
    DatasetIntelligenceStageTiming,
    DatasetIntelligenceUsage,
)
from .document_intelligence import (
    DocumentMetadata,
    OutputOptions,
    PipelineOptions,
    PredictOptions,
    ProcessDocumentResponse,
    RefinementStats,
    RefineOptions,
)
from .embed import EmbedType, UnifiedEmbedResponse
from .embedding import EmbedResponse
from .enrichment import (
    EnrichData,
    EnrichResponse,
)
from .extraction import ExtractionConfig, ExtractResponse
from .jobs import (
    CreditsResponse,
    JobCancelResponse,
    JobListResponse,
    JobStatus,
    JobStatusResponse,
    JobSubmittedResponse,
)
from .ontology import (
    BuildGraphResponse,
    EntityInput,
    OntologyConfig,
    OntologyEntityRef,
    OntologyRelation,
)
from .pipeline import (
    FileInput,
    PipelineConfig,
    PipelineExecutionSummary,
    PipelineInput,
    PipelineReport,  # noqa: F401
    PipelineResultResponse,
    PipelineStatusResponse,
    PipelineSubmitResponse,
    PipelineValidationResult,
    ServiceConfig,
    ServiceName,
    StageDownload,  # noqa: F401
    StageResult,
    StageStatus,
)
from .redaction import DetectPIIResponse, RedactionConfig

__all__ = [
    # Common
    "BaseResponse",
    "CustomLabel",
    "Entity",
    "KnowledgeGraph",
    "Message",
    "Relation",
    "Usage",
    # Unified Embed
    "EmbedType",
    "UnifiedEmbedResponse",
    # Embedding (legacy)
    "EmbedResponse",
    # ColBERT (legacy)
    "ColBERTEmbedResponse",
    # ColPali (legacy)
    "ColPaliEmbedResponse",
    # Compression
    "CompressResponse",
    "CompressMessagesResponse",
    # Enrichment
    "ChunkData",
    "ChunkItem",
    "ChunkResponse",
    "EnrichData",
    "EnrichResponse",
    # Document Intelligence
    "DocumentMetadata",
    "OutputOptions",
    "PipelineOptions",
    "PredictOptions",
    "ProcessDocumentResponse",
    "RefineOptions",
    "RefinementStats",
    # Extraction
    "ExtractionConfig",
    "ExtractResponse",
    # Ontology
    "BuildGraphResponse",
    "EntityInput",
    "OntologyConfig",
    "OntologyEntityRef",
    "OntologyRelation",
    # Pipeline
    "FileInput",
    "PipelineConfig",
    "PipelineExecutionSummary",
    "PipelineInput",
    "PipelineResultResponse",
    "PipelineStatusResponse",
    "PipelineSubmitResponse",
    "PipelineValidationResult",
    "ServiceConfig",
    "ServiceName",
    "StageResult",
    "StageStatus",
    # Redaction
    "DetectPIIResponse",
    "RedactionConfig",
    # Dataset Intelligence
    "DatasetIntelligenceDeltaSummary",
    "DatasetIntelligenceResponse",
    "DatasetIntelligenceStageTiming",
    "DatasetIntelligenceUsage",
    # Jobs
    "CreditsResponse",
    "JobCancelResponse",
    "JobListResponse",
    "JobStatus",
    "JobStatusResponse",
    "JobSubmittedResponse",
]
