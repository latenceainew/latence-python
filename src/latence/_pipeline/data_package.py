"""DataPackage: structured, composed result from a Data Intelligence Pipeline.

The DataPackage is the primary output a user receives from Latence.
It composes raw pipeline stage outputs into organized, summarized sections
ready for downstream use in RAG, agents, or LLM workflows.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .._models.common import Entity
from .._models.ontology import OntologyEntityRef, OntologyRelation
from .._models.pipeline import (
    PipelineExecutionSummary,
    PipelineResultResponse,
    StageResult,
)

_log = logging.getLogger("latence.pipeline")

# =============================================================================
# Section Models
# =============================================================================


class DocumentMetadataInfo(BaseModel):
    """Metadata about the processed document."""

    model_config = ConfigDict(extra="allow")

    filename: str | None = Field(default=None, description="Original filename")
    pages_processed: int | None = Field(default=None, description="Number of pages processed")
    content_type: str | None = Field(
        default=None, description="Content type (markdown, json, html)"
    )
    processing_mode: str | None = Field(default=None, description="Processing mode used")


class DocumentSection(BaseModel):
    """Processed document content from OCR/Document Intelligence.

    Contains the full extracted markdown text, optional per-page breakdowns,
    and document metadata.
    """

    model_config = ConfigDict(extra="allow")

    markdown: str = Field(description="Full clean markdown text")
    pages: list[str] | None = Field(
        default=None, description="Per-page markdown content (if available)"
    )
    metadata: DocumentMetadataInfo = Field(
        default_factory=DocumentMetadataInfo,
        description="Document processing metadata",
    )


class EntitySummary(BaseModel):
    """Summary statistics for extracted entities."""

    total: int = Field(default=0, description="Total number of entities")
    by_type: dict[str, int] = Field(default_factory=dict, description="Entity count by type")
    unique_labels: list[str] = Field(
        default_factory=list, description="All unique entity types found"
    )
    avg_confidence: float | None = Field(
        default=None, description="Average confidence score across all entities"
    )


class EntitiesSection(BaseModel):
    """Extracted entities with summary statistics.

    Provides both the full entity list and a computed summary with
    counts by type, unique labels, and average confidence.
    """

    model_config = ConfigDict(extra="allow")

    items: list[Entity] = Field(default_factory=list, description="Full entity list")
    summary: EntitySummary = Field(
        default_factory=EntitySummary, description="Entity summary statistics"
    )


class GraphSummary(BaseModel):
    """Summary statistics for a knowledge graph."""

    total_entities: int = Field(default=0, description="Total entities in graph")
    total_relations: int = Field(default=0, description="Total relations in graph")
    resolved_entities: int | None = Field(default=None, description="Entities after resolution")
    entity_types: list[str] = Field(default_factory=list, description="Unique entity type labels")
    relation_types: list[str] = Field(
        default_factory=list, description="Unique relation type labels"
    )


class KnowledgeGraphSection(BaseModel):
    """Knowledge graph with entities, relations, and summary.

    Contains the full graph data and computed summary statistics
    including type distributions.
    """

    model_config = ConfigDict(extra="allow")

    entities: list[Entity] = Field(default_factory=list, description="Resolved entities")
    relations: list[OntologyRelation] = Field(
        default_factory=list, description="Extracted relations"
    )
    summary: GraphSummary = Field(
        default_factory=GraphSummary, description="Graph summary statistics"
    )
    raw_graph: dict[str, Any] | str | None = Field(
        default=None,
        description="Raw knowledge graph output (KG dict or RDF string)",
    )


class RedactionSummary(BaseModel):
    """Summary statistics for PII redaction."""

    total_pii: int = Field(default=0, description="Total PII entities detected")
    by_type: dict[str, int] = Field(default_factory=dict, description="PII count by type")
    unique_labels: list[str] = Field(default_factory=list, description="Types of PII found")


class RedactionSection(BaseModel):
    """Redaction results: cleaned text and PII detection summary.

    Present only when redaction was included in the pipeline.
    """

    model_config = ConfigDict(extra="allow")

    redacted_text: str = Field(description="Text after PII redaction")
    pii_detected: list[Entity] = Field(default_factory=list, description="Detected PII entities")
    summary: RedactionSummary = Field(
        default_factory=RedactionSummary, description="Redaction summary statistics"
    )


class CompressionSummary(BaseModel):
    """Summary statistics for text compression."""

    compression_ratio: float = Field(
        default=0.0, description="Compression ratio (0-1, lower = more compressed)"
    )
    original_tokens: int = Field(default=0, description="Token count before compression")
    compressed_tokens: int = Field(default=0, description="Token count after compression")
    tokens_saved: int = Field(default=0, description="Tokens saved by compression")


class CompressionSection(BaseModel):
    """Compressed text output from the compression service.

    Present only when compression was included in the pipeline.
    """

    model_config = ConfigDict(extra="allow")

    compressed_text: str = Field(description="Text after compression")
    summary: CompressionSummary = Field(
        default_factory=CompressionSummary, description="Compression summary statistics"
    )


class ChunkingSummary(BaseModel):
    """Summary statistics for chunking results."""

    num_chunks: int = Field(default=0, description="Number of chunks produced")
    strategy: str = Field(default="hybrid", description="Chunking strategy used")
    chunk_size: int = Field(default=512, description="Target chunk size parameter")
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")


class ChunkingSection(BaseModel):
    """Chunking results: split text into semantically meaningful chunks.

    Present only when chunking was included in the pipeline.
    """

    model_config = ConfigDict(extra="allow")

    chunks: list[dict[str, Any]] = Field(
        default_factory=list, description="Chunk objects with metadata"
    )
    summary: ChunkingSummary = Field(
        default_factory=ChunkingSummary, description="Chunking summary statistics"
    )


class EnrichmentSummary(BaseModel):
    """Summary statistics for enrichment results."""

    num_chunks: int = Field(default=0, description="Number of chunks produced")
    strategy: str = Field(default="hybrid", description="Chunking strategy used")
    embedding_dim: int = Field(default=0, description="Embedding dimension")
    features_computed: list[str] = Field(
        default_factory=list, description="Feature groups that were computed"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")


class EnrichmentSection(BaseModel):
    """Enrichment results: chunks, embeddings, and feature data.

    Present only when enrichment was included in the pipeline.
    """

    model_config = ConfigDict(extra="allow")

    chunks: list[dict[str, Any]] = Field(
        default_factory=list, description="Chunk objects with metadata"
    )
    embeddings: list[Any] = Field(default_factory=list, description="Chunk embeddings")
    features: dict[str, Any] = Field(
        default_factory=dict, description="Feature groups keyed by name"
    )
    summary: EnrichmentSummary = Field(
        default_factory=EnrichmentSummary, description="Enrichment summary statistics"
    )


class StageReport(BaseModel):
    """Report for a single pipeline stage."""

    service: str = Field(description="Service name")
    status: str = Field(description="Stage status")
    processing_time_ms: float | None = Field(default=None, description="Processing time in ms")
    credits_used: float | None = Field(default=None, description="Credits consumed by this stage")
    items_produced: int | None = Field(
        default=None, description="Number of items produced (entities, relations, etc.)"
    )
    error: str | None = Field(default=None, description="Error message if failed")


class ConfidenceScores(BaseModel):
    """Confidence metrics across the pipeline."""

    ocr_quality: float | None = Field(
        default=None, description="OCR/document processing quality estimate"
    )
    entity_avg_confidence: float | None = Field(
        default=None, description="Average entity extraction confidence"
    )
    graph_completeness: float | None = Field(
        default=None, description="Knowledge graph completeness estimate"
    )


class QualityReport(BaseModel):
    """Processing quality report and confidence metrics.

    Always present in a DataPackage. Provides per-stage timing,
    aggregate confidence scores, and cost tracking.
    """

    stages: list[StageReport] = Field(
        default_factory=list, description="Per-stage processing reports"
    )
    confidence: ConfidenceScores = Field(
        default_factory=ConfidenceScores, description="Confidence metrics"
    )
    total_processing_time_ms: float = Field(
        default=0.0, description="Total processing time across all stages"
    )
    total_cost_usd: float | None = Field(default=None, description="Total cost in USD")


# =============================================================================
# DataPackage
# =============================================================================


class DataPackage(BaseModel):
    """Composed, structured result from a Data Intelligence Pipeline.

    This is the primary output a user receives from Latence.
    Every section is cleanly organized, summarized, and ready for
    downstream use in RAG, agents, or LLM workflows.

    Sections are present based on which pipeline steps were executed:
    - ``document``: from OCR / Document Intelligence
    - ``entities``: from Entity Extraction
    - ``knowledge_graph``: from Relation Extraction / Knowledge Graph construction
    - ``redaction``: from PII Redaction (only if enabled)
    - ``compression``: from Text Compression (only if enabled)
    - ``quality``: always present -- processing report + confidence scores

    Example:
        >>> pkg = job.wait_for_completion()
        >>> print(pkg.document.markdown)
        >>> print(pkg.entities.summary.total)
        >>> print(pkg.knowledge_graph.summary.total_relations)
        >>> print(pkg.compression.summary.compression_ratio)
        >>> pkg.download_archive("./results.zip")
    """

    model_config = ConfigDict(extra="allow")

    # Identity
    id: str = Field(description="Pipeline job ID")
    name: str | None = Field(default=None, description="Pipeline name")
    created_at: str = Field(description="Job creation timestamp")
    status: str = Field(description="Final job status")

    # Content sections (present based on which steps ran)
    document: DocumentSection | None = Field(
        default=None, description="OCR / Document Intelligence output"
    )
    entities: EntitiesSection | None = Field(default=None, description="Entity Extraction output")
    knowledge_graph: KnowledgeGraphSection | None = Field(
        default=None, description="Knowledge Graph / Relation Extraction output"
    )
    redaction: RedactionSection | None = Field(
        default=None, description="PII Redaction output (if enabled)"
    )
    compression: CompressionSection | None = Field(
        default=None, description="Text Compression output (if enabled)"
    )
    chunking: ChunkingSection | None = Field(
        default=None, description="Chunking output — split text into chunks (if enabled)"
    )
    enrichment: EnrichmentSection | None = Field(
        default=None, description="Enrichment output — chunks, embeddings, features (Coming Soon)"
    )

    # Always present
    quality: QualityReport = Field(
        default_factory=QualityReport,
        description="Processing report + confidence metrics",
    )

    # Raw access (for power users)
    raw: dict[str, Any] | None = Field(
        default=None, description="Original unprocessed API response"
    )

    _parse_warnings: list[str] = []

    @property
    def parse_warnings(self) -> list[str]:
        """Warnings from parsing malformed items in the pipeline output."""
        return list(self._parse_warnings)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pipeline_result(
        cls,
        result: PipelineResultResponse,
        *,
        name: str | None = None,
        services: list[str] | None = None,
    ) -> DataPackage:
        """Compose a DataPackage from a raw PipelineResultResponse.

        This is the main composition entry point.  It inspects which
        stages were executed, extracts and organizes each section,
        and computes summary statistics.

        Args:
            result: Raw pipeline result from the API.
            name: Pipeline name (passed through from submission).
            services: Ordered list of services that were executed.
        """
        raw_data: dict[str, Any] = {}
        if result.final_output:
            raw_data["final_output"] = result.final_output
        if result.intermediate_results:
            raw_data["intermediate_results"] = {
                k: v.model_dump() if isinstance(v, BaseModel) else v
                for k, v in result.intermediate_results.items()
            }

        # Gather stage outputs keyed by service name
        stage_outputs = _collect_stage_outputs(result)

        # Build sections
        document = _build_document_section(stage_outputs)
        entities_section = _build_entities_section(stage_outputs)
        kg_section, kg_warnings = _build_knowledge_graph_section(stage_outputs)
        redaction_section = _build_redaction_section(stage_outputs)
        compression_section = _build_compression_section(stage_outputs)
        chunking_section = _build_chunking_section(stage_outputs)
        enrichment_section = _build_enrichment_section(stage_outputs)

        # Build quality report
        quality = _build_quality_report(
            result.execution_summary,
            result.intermediate_results,
            entities_section,
            kg_section,
            document,
            services,
        )

        server_created_at = getattr(result, "created_at", None)
        created_at = (
            server_created_at if server_created_at else datetime.now(timezone.utc).isoformat()
        )

        pkg = cls(
            id=result.job_id,
            name=name,
            created_at=created_at,
            status=result.status,
            document=document,
            entities=entities_section,
            knowledge_graph=kg_section,
            redaction=redaction_section,
            compression=compression_section,
            chunking=chunking_section,
            enrichment=enrichment_section,
            quality=quality,
            raw=raw_data or None,
        )
        pkg._parse_warnings = kg_warnings
        return pkg

    # ------------------------------------------------------------------
    # Archive
    # ------------------------------------------------------------------

    def download_archive(self, path: str | Path) -> Path:
        """Download an organized ZIP archive of all results.

        Creates a ZIP file with the following structure::

            {name}/
              README.md
              document.md
              entities.json
              knowledge_graph.json
              redaction.json        (if present)
              quality_report.json
              metadata.json
              pages/
                page_001.md
                page_002.md
                ...

        Args:
            path: Destination file path for the ZIP archive.

        Returns:
            Path to the created ZIP file.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        folder_name = _sanitize_name(self.name) if self.name else f"latence_{self.id}"

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # README
            zf.writestr(f"{folder_name}/README.md", self._generate_readme())

            # Document
            if self.document:
                zf.writestr(f"{folder_name}/document.md", self.document.markdown)
                if self.document.pages:
                    for i, page in enumerate(self.document.pages, 1):
                        zf.writestr(f"{folder_name}/pages/page_{i:03d}.md", page)

            # Entities
            if self.entities:
                zf.writestr(
                    f"{folder_name}/entities.json",
                    _to_json(self.entities.model_dump()),
                )

            # Knowledge Graph
            if self.knowledge_graph:
                zf.writestr(
                    f"{folder_name}/knowledge_graph.json",
                    _to_json(self.knowledge_graph.model_dump()),
                )

            # Redaction
            if self.redaction:
                zf.writestr(
                    f"{folder_name}/redaction.json",
                    _to_json(self.redaction.model_dump()),
                )

            # Compression
            if self.compression:
                zf.writestr(
                    f"{folder_name}/compression.json",
                    _to_json(self.compression.model_dump()),
                )

            # Chunking
            if self.chunking:
                zf.writestr(
                    f"{folder_name}/chunking.json",
                    _to_json(self.chunking.model_dump()),
                )

            # Enrichment
            if self.enrichment:
                zf.writestr(
                    f"{folder_name}/enrichment.json",
                    _to_json(self.enrichment.model_dump()),
                )

            # Quality Report
            zf.writestr(
                f"{folder_name}/quality_report.json",
                _to_json(self.quality.model_dump()),
            )

            # Metadata
            metadata = {
                "id": self.id,
                "name": self.name,
                "created_at": self.created_at,
                "status": self.status,
                "sections_present": [
                    s
                    for s, v in [
                        ("document", self.document),
                        ("entities", self.entities),
                        ("knowledge_graph", self.knowledge_graph),
                        ("redaction", self.redaction),
                        ("compression", self.compression),
                        ("chunking", self.chunking),
                        ("enrichment", self.enrichment),
                    ]
                    if v is not None
                ],
            }
            zf.writestr(f"{folder_name}/metadata.json", _to_json(metadata))

        dest.write_bytes(buf.getvalue())
        return dest

    # ------------------------------------------------------------------
    # Merge (client-side consolidation)
    # ------------------------------------------------------------------

    def merge(
        self,
        *,
        save_to: str | Path | None = None,
        indent: int = 2,
    ) -> dict[str, Any]:
        """Merge all pipeline outputs into a single, document-centric JSON.

        This is a **pure client-side** convenience that reorganizes the
        already-downloaded DataPackage into a flat, redundancy-free dict
        optimized for developer consumption.

        Design principles:
          - **Document-centric**: all outputs grouped per document, not per service
          - **Zero redundancy**: markdown text appears once under
            ``documents[].markdown``, not repeated in extraction/ontology output
          - **Flat and navigable**: ``doc["entities"]`` instead of
            ``pkg.entities.items``
          - **Opt-in sections**: only includes services that actually ran
          - **Single-file shortcut**: for single-document pipelines,
            ``merge()["documents"][0]`` gives everything

        Args:
            save_to: If provided, write the JSON to this file path.
            indent: JSON indentation level (default 2).

        Returns:
            Document-centric dict ready for serialization or direct use.

        Example::

            pkg = job.wait_for_completion()
            unified = pkg.merge()
            print(unified["documents"][0]["entities"][0]["text"])

            # Or save directly:
            pkg.merge(save_to="./results.json")
        """
        # --- Build per-document entries ---
        documents: list[dict[str, Any]] = []

        # For single-doc pipelines (most common), create one document entry
        doc_entry: dict[str, Any] = {}

        if self.document:
            doc_entry["filename"] = self.document.metadata.filename or "document"
            doc_entry["markdown"] = self.document.markdown
            if self.document.pages:
                doc_entry["pages"] = self.document.pages
            doc_entry["page_count"] = (
                len(self.document.pages)
                if self.document.pages
                else self.document.metadata.pages_processed or 0
            )
        else:
            doc_entry["filename"] = "input"
            doc_entry["markdown"] = ""
            doc_entry["page_count"] = 0

        if self.entities:
            doc_entry["entities"] = [
                {
                    "text": e.text,
                    "label": e.label,
                    "score": e.score,
                    "start": e.start,
                    "end": e.end,
                    "source": e.source,
                }
                for e in self.entities.items
            ]

        if self.knowledge_graph:
            kg: dict[str, Any] = {}
            if self.knowledge_graph.entities:
                kg["entities"] = [
                    {
                        "text": e.text,
                        "label": e.label,
                        "score": e.score,
                    }
                    for e in self.knowledge_graph.entities
                ]
            if self.knowledge_graph.relations:
                kg["relations"] = [
                    {
                        "source": r.entity1.text
                        if isinstance(r.entity1, OntologyEntityRef)
                        else str(r.entity1),
                        "relation": r.relation_type,
                        "target": r.entity2.text
                        if isinstance(r.entity2, OntologyEntityRef)
                        else str(r.entity2),
                        "score": r.score,
                    }
                    for r in self.knowledge_graph.relations
                ]
            if kg:
                doc_entry["knowledge_graph"] = kg

        if self.redaction:
            doc_entry["redaction"] = {
                "redacted_text": self.redaction.redacted_text,
                "pii_items": [
                    {
                        "text": e.text,
                        "label": e.label,
                        "score": e.score,
                        "start": e.start,
                        "end": e.end,
                    }
                    for e in self.redaction.pii_detected
                ],
            }

        if self.compression:
            doc_entry["compression"] = {
                "text": self.compression.compressed_text,
                "ratio": self.compression.summary.compression_ratio,
                "tokens_saved": self.compression.summary.tokens_saved,
            }

        if self.chunking:
            doc_entry["chunking"] = {
                "num_chunks": self.chunking.summary.num_chunks,
                "strategy": self.chunking.summary.strategy,
                "chunks": self.chunking.chunks,
            }

        if self.enrichment:
            doc_entry["enrichment"] = {
                "num_chunks": self.enrichment.summary.num_chunks,
                "strategy": self.enrichment.summary.strategy,
                "features_computed": self.enrichment.summary.features_computed,
                "chunks": self.enrichment.chunks,
                "features": self.enrichment.features,
            }

        documents.append(doc_entry)

        # --- Summary ---
        entity_by_type: dict[str, int] = {}
        relation_by_type: dict[str, int] = {}

        if self.entities:
            entity_by_type = dict(self.entities.summary.by_type)
        if self.knowledge_graph:
            for r in self.knowledge_graph.relations:
                rt = r.relation_type
                relation_by_type[rt] = relation_by_type.get(rt, 0) + 1

        services_executed = [s.service for s in self.quality.stages if s.status == "completed"]

        summary: dict[str, Any] = {
            "documents": len(documents),
            "pages": sum(d.get("page_count", 0) for d in documents),
            "entities": {
                "total": self.entities.summary.total if self.entities else 0,
                "by_type": entity_by_type,
            },
            "relations": {
                "total": self.knowledge_graph.summary.total_relations
                if self.knowledge_graph
                else 0,
                "by_type": relation_by_type,
            },
            "cost_usd": self.quality.total_cost_usd,
            "processing_time_ms": self.quality.total_processing_time_ms,
            "services_executed": services_executed,
        }

        merged: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "documents": documents,
            "summary": summary,
        }

        if save_to is not None:
            dest = Path(save_to)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(merged, indent=indent, default=str), encoding="utf-8")

        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_readme(self) -> str:
        """Generate a human-readable README for the archive."""
        lines = [
            f"# Data Package: {self.name or self.id}",
            "",
            f"**Job ID:** {self.id}",
            f"**Status:** {self.status}",
            f"**Created:** {self.created_at}",
            "",
            "## Contents",
            "",
        ]

        if self.document:
            pages = self.document.metadata.pages_processed or "N/A"
            lines.append(f"- **document.md** -- Extracted markdown ({pages} pages)")

        if self.entities:
            total = self.entities.summary.total
            types = ", ".join(self.entities.summary.unique_labels[:5])
            lines.append(f"- **entities.json** -- {total} entities ({types})")

        if self.knowledge_graph:
            ents = self.knowledge_graph.summary.total_entities
            rels = self.knowledge_graph.summary.total_relations
            lines.append(f"- **knowledge_graph.json** -- {ents} entities, {rels} relations")

        if self.redaction:
            pii = self.redaction.summary.total_pii
            lines.append(f"- **redaction.json** -- {pii} PII entities redacted")

        if self.compression:
            ratio = self.compression.summary.compression_ratio
            saved = self.compression.summary.tokens_saved
            lines.append(
                f"- **compression.json** -- {ratio:.0%} compression ratio, {saved} tokens saved"
            )

        if self.chunking:
            nc = self.chunking.summary.num_chunks
            strat = self.chunking.summary.strategy
            lines.append(f"- **chunking.json** -- {nc} chunks, strategy: {strat}")

        if self.enrichment:
            nc = self.enrichment.summary.num_chunks
            feats = ", ".join(self.enrichment.summary.features_computed[:5])
            lines.append(f"- **enrichment.json** -- {nc} chunks, features: {feats}")

        lines.extend(
            [
                "- **quality_report.json** -- Processing report & confidence",
                "- **metadata.json** -- Pipeline configuration & timing",
                "",
                "## Quality",
                "",
                f"- Total processing time: {self.quality.total_processing_time_ms:.0f}ms",
            ]
        )

        if self.quality.confidence.ocr_quality is not None:
            lines.append(f"- OCR quality: {self.quality.confidence.ocr_quality:.2f}")
        if self.quality.confidence.entity_avg_confidence is not None:
            lines.append(
                f"- Entity avg confidence: {self.quality.confidence.entity_avg_confidence:.2f}"
            )
        if self.quality.confidence.graph_completeness is not None:
            lines.append(f"- Graph completeness: {self.quality.confidence.graph_completeness:.2f}")

        lines.extend(
            [
                "",
                "---",
                "*Generated by Latence Data Intelligence Pipeline*",
            ]
        )
        return "\n".join(lines)


# =============================================================================
# Internal composition helpers
# =============================================================================


def _collect_stage_outputs(result: PipelineResultResponse) -> dict[str, dict[str, Any]]:
    """Collect stage outputs keyed by service name.

    Prefers intermediate_results (richer per-stage data) and falls back
    to final_output.
    """
    outputs: dict[str, dict[str, Any]] = {}

    if result.intermediate_results:
        for service_name, stage in result.intermediate_results.items():
            if isinstance(stage, StageResult):
                if stage.output:
                    outputs[service_name] = stage.output
            elif isinstance(stage, dict):
                output = stage.get("output")
                if output:
                    outputs[service_name] = output

    # Final output is the output of the *last* stage
    if result.final_output:
        outputs["_final"] = result.final_output

    return outputs


def _build_document_section(
    stage_outputs: dict[str, dict[str, Any]],
) -> DocumentSection | None:
    """Build DocumentSection from document_intelligence output."""
    data = stage_outputs.get("document_intelligence")
    if not data:
        return None

    markdown = data.get("content", "")
    if not markdown and not data.get("pages"):
        return None

    pages: list[str] | None = None
    raw_pages = data.get("pages")
    if raw_pages and isinstance(raw_pages, list):
        pages = []
        for p in raw_pages:
            if isinstance(p, dict):
                pages.append(p.get("markdown", p.get("content", str(p))))  # type: ignore[arg-type]
            else:
                pages.append(str(p))

    metadata_raw = data.get("metadata", {})
    metadata = DocumentMetadataInfo(
        filename=metadata_raw.get("filename") if isinstance(metadata_raw, dict) else None,
        pages_processed=data.get("pages_processed"),
        content_type=data.get("content_type"),
        processing_mode=metadata_raw.get("processing_mode")
        if isinstance(metadata_raw, dict)
        else None,
    )

    return DocumentSection(markdown=markdown, pages=pages, metadata=metadata)


def _build_entities_section(
    stage_outputs: dict[str, dict[str, Any]],
) -> EntitiesSection | None:
    """Build EntitiesSection from extraction output."""
    data = stage_outputs.get("extraction")
    if not data:
        return None

    # RunPod services may nest output under "result" or "data"
    inner = data.get("result", data.get("data", data))
    raw_entities = (
        inner.get("entities", []) if isinstance(inner, dict) else data.get("entities", [])
    )
    items: list[Entity] = []
    for e in raw_entities:
        if isinstance(e, Entity):
            items.append(e)
        elif isinstance(e, dict):
            items.append(Entity.model_validate(e))

    if not items:
        return None

    # Compute summary
    type_counts: Counter[str] = Counter(e.label for e in items)
    scores = [e.score for e in items if e.score is not None]
    avg_conf = sum(scores) / len(scores) if scores else None

    summary = EntitySummary(
        total=len(items),
        by_type=dict(type_counts),
        unique_labels=sorted(type_counts.keys()),
        avg_confidence=round(avg_conf, 4) if avg_conf is not None else None,
    )

    return EntitiesSection(items=items, summary=summary)


def _build_knowledge_graph_section(
    stage_outputs: dict[str, dict[str, Any]],
) -> tuple[KnowledgeGraphSection | None, list[str]]:
    """Build KnowledgeGraphSection from ontology output.

    Returns:
        Tuple of (section_or_None, parse_warnings).
    """
    data = stage_outputs.get("ontology")
    if not data:
        return None, []

    # RunPod services may nest output under "data" or "result"
    inner = data.get("data", data.get("result", data))

    # Entities
    raw_entities = inner.get("entities", []) if isinstance(inner, dict) else []
    entities: list[Entity] = []
    for e in raw_entities:
        if isinstance(e, Entity):
            entities.append(e)
        elif isinstance(e, dict):
            entities.append(Entity.model_validate(e))

    # Relations
    raw_relations = inner.get("relations", []) if isinstance(inner, dict) else []
    relations: list[OntologyRelation] = []
    parse_warnings: list[str] = []
    for r in raw_relations:
        if isinstance(r, OntologyRelation):
            relations.append(r)
        elif isinstance(r, dict):
            try:
                relations.append(OntologyRelation.model_validate(r))
            except Exception as exc:
                msg = f"Skipped malformed relation: {exc}"
                _log.warning(msg)
                parse_warnings.append(msg)

    # Raw graph
    raw_graph = inner.get("knowledge_graph") if isinstance(inner, dict) else None

    # Summary
    entity_types = sorted({e.label for e in entities})
    relation_types = sorted({r.relation_type for r in relations})

    summary = GraphSummary(
        total_entities=len(entities),
        total_relations=len(relations),
        resolved_entities=inner.get("resolved_entities") if isinstance(inner, dict) else None,
        entity_types=entity_types,
        relation_types=relation_types,
    )

    return KnowledgeGraphSection(
        entities=entities,
        relations=relations,
        summary=summary,
        raw_graph=raw_graph,
    ), parse_warnings


def _build_redaction_section(
    stage_outputs: dict[str, dict[str, Any]],
) -> RedactionSection | None:
    """Build RedactionSection from redaction output."""
    data = stage_outputs.get("redaction")
    if not data:
        return None

    # RunPod services may nest output under "result" or "data"
    inner = data.get("result", data.get("data", data))

    redacted_text = (
        inner.get("redacted_text", "") if isinstance(inner, dict) else data.get("redacted_text", "")
    )
    if not redacted_text:
        return None

    raw_entities = (
        inner.get("entities", []) if isinstance(inner, dict) else data.get("entities", [])
    )
    pii_detected: list[Entity] = []
    for e in raw_entities:
        if isinstance(e, Entity):
            pii_detected.append(e)
        elif isinstance(e, dict):
            pii_detected.append(Entity.model_validate(e))

    type_counts: Counter[str] = Counter(e.label for e in pii_detected)

    summary = RedactionSummary(
        total_pii=len(pii_detected),
        by_type=dict(type_counts),
        unique_labels=sorted(type_counts.keys()),
    )

    return RedactionSection(
        redacted_text=redacted_text,
        pii_detected=pii_detected,
        summary=summary,
    )


def _build_compression_section(
    stage_outputs: dict[str, dict[str, Any]],
) -> CompressionSection | None:
    """Build CompressionSection from compression output."""
    data = stage_outputs.get("compression")
    if not data:
        return None

    # RunPod services may nest output under "result" or "data"
    inner = data.get("result", data.get("data", data))

    compressed_text = (
        inner.get("compressed_text", "")
        if isinstance(inner, dict)
        else data.get("compressed_text", "")
    )
    if not compressed_text:
        return None

    original_tokens = (
        inner.get("original_tokens", 0)
        if isinstance(inner, dict)
        else data.get("original_tokens", 0)
    )
    compressed_tokens = (
        inner.get("compressed_tokens", 0)
        if isinstance(inner, dict)
        else data.get("compressed_tokens", 0)
    )
    tokens_saved = original_tokens - compressed_tokens if original_tokens else 0
    compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0.0

    summary = CompressionSummary(
        compression_ratio=round(compression_ratio, 4),
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        tokens_saved=tokens_saved,
    )

    return CompressionSection(compressed_text=compressed_text, summary=summary)


def _build_chunking_section(
    stage_outputs: dict[str, dict[str, Any]],
) -> ChunkingSection | None:
    """Build ChunkingSection from chunking output."""
    data = stage_outputs.get("chunking")
    if not data:
        return None

    inner = data.get("result", data.get("data", data))

    chunks = inner.get("chunks", []) if isinstance(inner, dict) else []
    num_chunks = inner.get("num_chunks", len(chunks)) if isinstance(inner, dict) else len(chunks)
    strategy = inner.get("strategy", "hybrid") if isinstance(inner, dict) else "hybrid"
    chunk_size = inner.get("chunk_size", 512) if isinstance(inner, dict) else 512
    processing_time_ms = inner.get("processing_time_ms", 0.0) if isinstance(inner, dict) else 0.0

    summary = ChunkingSummary(
        num_chunks=num_chunks,
        strategy=strategy,
        chunk_size=chunk_size,
        processing_time_ms=processing_time_ms,
    )

    return ChunkingSection(chunks=chunks, summary=summary)


def _build_enrichment_section(
    stage_outputs: dict[str, dict[str, Any]],
) -> EnrichmentSection | None:
    """Build EnrichmentSection from enrichment output."""
    data = stage_outputs.get("enrichment")
    if not data:
        return None

    inner = data.get("result", data.get("data", data))

    chunks = inner.get("chunks", []) if isinstance(inner, dict) else []
    embeddings = inner.get("embeddings", []) if isinstance(inner, dict) else []
    features = inner.get("features", {}) if isinstance(inner, dict) else {}
    num_chunks = inner.get("num_chunks", len(chunks)) if isinstance(inner, dict) else len(chunks)
    strategy = inner.get("strategy", "hybrid") if isinstance(inner, dict) else "hybrid"
    embedding_dim = inner.get("embedding_dim", 0) if isinstance(inner, dict) else 0
    processing_time_ms = inner.get("processing_time_ms", 0.0) if isinstance(inner, dict) else 0.0

    features_computed = list(features.keys()) if isinstance(features, dict) else []

    summary = EnrichmentSummary(
        num_chunks=num_chunks,
        strategy=strategy,
        embedding_dim=embedding_dim,
        features_computed=features_computed,
        processing_time_ms=processing_time_ms,
    )

    return EnrichmentSection(
        chunks=chunks,
        embeddings=embeddings,
        features=features,
        summary=summary,
    )


def _build_quality_report(
    execution_summary: PipelineExecutionSummary,
    intermediate_results: dict[str, StageResult] | None,
    entities_section: EntitiesSection | None,
    kg_section: KnowledgeGraphSection | None,
    document_section: DocumentSection | None,
    services: list[str] | None,
) -> QualityReport:
    """Build QualityReport from execution summary and stage data."""
    stages: list[StageReport] = []

    if intermediate_results:
        for service_name, stage in intermediate_results.items():
            if isinstance(stage, StageResult):
                items_produced = _count_items(service_name, stage.output)
                stages.append(
                    StageReport(
                        service=service_name,
                        status=stage.status,
                        processing_time_ms=stage.processing_time_ms,
                        credits_used=stage.credits_used,
                        items_produced=items_produced,
                        error=stage.error,
                    )
                )
            elif isinstance(stage, dict):
                output = stage.get("output")
                items_produced = _count_items(service_name, output)
                stages.append(
                    StageReport(
                        service=service_name,
                        status=stage.get("status", "unknown"),
                        processing_time_ms=stage.get("processing_time_ms"),
                        credits_used=stage.get("credits_used"),
                        items_produced=items_produced,
                        error=stage.get("error"),
                    )
                )
    elif services:
        # If no intermediate results, create stub reports from services list
        for svc in services:
            stages.append(
                StageReport(
                    service=svc,
                    status="completed",
                )
            )

    # Confidence scores
    confidence = ConfidenceScores()

    if entities_section and entities_section.summary.avg_confidence is not None:
        confidence.entity_avg_confidence = entities_section.summary.avg_confidence

    if kg_section:
        # Estimate graph completeness from relation/entity ratio
        if kg_section.summary.total_entities > 0:
            ratio = min(
                kg_section.summary.total_relations / kg_section.summary.total_entities,
                1.0,
            )
            confidence.graph_completeness = round(ratio, 4)

    return QualityReport(
        stages=stages,
        confidence=confidence,
        total_processing_time_ms=execution_summary.total_processing_time_ms,
        total_cost_usd=execution_summary.total_credits_used or None,
    )


def _count_items(service_name: str, output: dict[str, Any] | None) -> int | None:
    """Count items produced by a stage."""
    if not output:
        return None
    if service_name == "extraction":
        entities = output.get("entities", [])
        return len(entities) if isinstance(entities, list) else None
    if service_name == "ontology":
        return output.get("relation_count") or output.get("entity_count")
    if service_name == "redaction":
        return output.get("entity_count")
    if service_name == "document_intelligence":
        return output.get("pages_processed")
    if service_name == "compression":
        return output.get("compressed_tokens")
    if service_name == "chunking":
        return output.get("num_chunks")
    if service_name == "enrichment":
        return output.get("num_chunks")
    return None


def _sanitize_name(name: str) -> str:
    """Sanitize a pipeline name for use as a folder name."""
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
    return safe.strip().replace(" ", "_")[:100]


def _to_json(data: Any) -> str:
    """Serialize to pretty JSON string."""
    return json.dumps(data, indent=2, default=str, ensure_ascii=False)
