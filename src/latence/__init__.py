"""
Latence AI Python SDK -- Data Intelligence Pipeline

Turn messy documents into RAG-ready knowledge with a single call.
The pipeline is async/job-based by default: submit files, get back a
structured DataPackage with document markdown, entities, knowledge
graphs, quality metrics, and more.

Simplest usage::

    >>> from latence import Latence
    >>> client = Latence(api_key="lat_xxx")
    >>> job = client.pipeline.run(files=["contract.pdf"])
    >>> pkg = job.wait_for_completion()
    >>> print(pkg.document.markdown)
    >>> print(pkg.entities.summary)
    >>> print(pkg.knowledge_graph.relations)
    >>> pkg.download_archive("./results.zip")

With explicit steps::

    >>> job = client.pipeline.run(
    ...     files=["contract.pdf"],
    ...     steps={
    ...         "ocr": {"mode": "performance"},
    ...         "redaction": {"mode": "balanced"},
    ...         "extraction": {"threshold": 0.3},
    ...         "knowledge_graph": {"resolve_entities": True},
    ...     },
    ...     name="Legal Contracts",
    ... )
    >>> pkg = job.wait_for_completion()

Fluent builder (power users)::

    >>> from latence import PipelineBuilder
    >>> config = (
    ...     PipelineBuilder()
    ...     .doc_intel(mode="performance")
    ...     .extraction(threshold=0.3)
    ...     .ontology(resolve_entities=True)
    ...     .build()
    ... )
    >>> job = client.pipeline.submit(config, files=["contract.pdf"])
    >>> pkg = job.wait_for_completion()

Experimental (self-service direct access)::

    >>> result = client.experimental.extraction.extract(text="Apple is in Cupertino.")

Async usage::

    >>> from latence import AsyncLatence
    >>> async with AsyncLatence(api_key="lat_xxx") as client:
    ...     job = await client.pipeline.run(files=["contract.pdf"])
    ...     pkg = await job.wait_for_completion()

Enable debug logging::

    >>> import latence
    >>> latence.setup_logging("DEBUG")
"""

from __future__ import annotations

# Version
from ._version import VERSION

__version__ = VERSION

# Main clients
from ._client import AsyncLatence, Latence

# Exceptions
from ._exceptions import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    InsufficientCreditsError,
    JobError,
    JobTimeoutError,
    LatenceError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TransportError,
    ValidationError,
)

# Logging
from ._logging import setup_logging

# Models - commonly used types
from ._models import (
    # Responses
    BuildGraphResponse,
    ChunkResponse,
    ColBERTEmbedResponse,
    ColPaliEmbedResponse,
    CompressMessagesResponse,
    CompressResponse,
    CreditsResponse,
    DetectPIIResponse,
    EmbedResponse,
    # Unified Embed
    EmbedType,
    EnrichResponse,
    # Common
    Entity,
    ExtractResponse,
    JobStatusResponse,
    JobSubmittedResponse,
    KnowledgeGraph,
    Message,
    # Pipeline
    PipelineConfig,
    PipelineReport,
    PipelineResultResponse,
    PipelineSubmitResponse,
    ProcessDocumentResponse,
    Relation,
    ServiceConfig,
    # Trace
    SessionState,
    StageDownload,
    StageStatus,
    SupportUnitInput,
    TraceCodeResponse,
    TraceRagResponse,
    TraceResponse,
    TraceRollupResponse,
    UnifiedEmbedResponse,
    Usage,
)

# Pipeline (primary interface)
from ._pipeline import (
    AsyncJob,
    DataPackage,
    Job,
    PipelineBuilder,
    PipelineValidationError,
)

__all__ = [
    # Version
    "__version__",
    # Main clients
    "Latence",
    "AsyncLatence",
    # Logging
    "setup_logging",
    # Exceptions
    "LatenceError",
    "APIError",
    "AuthenticationError",
    "InsufficientCreditsError",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "ServerError",
    "TransportError",
    "APIConnectionError",
    "APITimeoutError",
    "JobError",
    "JobTimeoutError",
    "PipelineValidationError",
    # Pipeline (primary interface)
    "Job",
    "AsyncJob",
    "DataPackage",
    # Common models
    "Entity",
    "KnowledgeGraph",
    "Message",
    "Relation",
    "Usage",
    # Unified Embed
    "EmbedType",
    "UnifiedEmbedResponse",
    # Response models
    "BuildGraphResponse",
    "ColBERTEmbedResponse",
    "ColPaliEmbedResponse",
    "ChunkResponse",
    "CompressMessagesResponse",
    "CompressResponse",
    "CreditsResponse",
    "EnrichResponse",
    "DetectPIIResponse",
    "EmbedResponse",
    "ExtractResponse",
    "JobStatusResponse",
    "JobSubmittedResponse",
    "ProcessDocumentResponse",
    # Pipeline models
    "PipelineBuilder",
    "PipelineConfig",
    "PipelineReport",
    "PipelineResultResponse",
    "PipelineSubmitResponse",
    "ServiceConfig",
    "StageDownload",
    "StageStatus",
    # Trace
    "SessionState",
    "SupportUnitInput",
    "TraceCodeResponse",
    "TraceRagResponse",
    "TraceResponse",
    "TraceRollupResponse",
]
