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
    PipelineReport,
    PipelineResultResponse,
    PipelineStatusResponse,
    PipelineSubmitResponse,
    PipelineValidationResult,
    ServiceConfig,
    ServiceName,
    StageDownload,
    StageResult,
    StageStatus,
)
from .redaction import DetectPIIResponse, RedactionConfig
from .trace import (
    AttributionMode,
    CodeLaneDiagnostics,
    FileAttribution,
    Heatmap,
    HeatmapFormat,
    PrimaryMetric,
    ResponseLanguageHint,
    ScoringMode,
    SegmentationMode,
    SessionSignals,
    SessionState,
    SupportUnitInput,
    SupportUnitUsage,
    SupportUnitsUsageSummary,
    TraceCodeResponse,
    TraceRagResponse,
    TraceResponse,
    TraceRollupResponse,
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
    "PipelineReport",
    "StageDownload",
    "StageResult",
    "StageStatus",
    # Redaction
    "DetectPIIResponse",
    "RedactionConfig",
    # Trace
    "AttributionMode",
    "CodeLaneDiagnostics",
    "FileAttribution",
    "Heatmap",
    "HeatmapFormat",
    "PrimaryMetric",
    "ResponseLanguageHint",
    "ScoringMode",
    "SegmentationMode",
    "SessionSignals",
    "SessionState",
    "SupportUnitInput",
    "SupportUnitUsage",
    "SupportUnitsUsageSummary",
    "TraceCodeResponse",
    "TraceRagResponse",
    "TraceResponse",
    "TraceRollupResponse",
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
