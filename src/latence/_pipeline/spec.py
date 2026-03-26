"""Pipeline specification: step aliases, smart defaults, ordering, and config parsing.

This module translates user-friendly pipeline configurations (dict-based
or simplified) into the internal ``ServiceConfig`` / ``PipelineConfig``
models the API gateway expects.

The service dependency graph mirrors the pipeline worker's
``config.SERVICE_PARENT`` -- the orchestrator topologically sorts services
and runs independent branches (e.g. extraction + redaction + compression)
in parallel execution waves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

from .._models.pipeline import FileInput, PipelineConfig, PipelineInput, ServiceConfig
from .._utils import file_to_base64

# =============================================================================
# Step Aliases
# =============================================================================

#: Mapping from friendly step names to internal service names.
#: All original service names also map to themselves.
STEP_ALIASES: dict[str, str] = {
    # Friendly aliases
    "ocr": "document_intelligence",
    "doc_intel": "document_intelligence",
    "knowledge_graph": "ontology",
    "graph": "ontology",
    "relation_extraction": "ontology",
    "redact": "redaction",
    "extract": "extraction",
    "compress": "compression",
    "enrich": "enrichment",
    "feature_enrichment": "enrichment",
    "resolve": "graph_ontology_builder",
    "graph_builder": "graph_ontology_builder",
    "graph_resolution": "graph_ontology_builder",
    "entity_linking": "graph_ontology_builder",
    "link": "graph_ontology_builder",
    "linking": "graph_ontology_builder",
    "glinker": "graph_ontology_builder",
    # Identity mappings (canonical service names)
    "document_intelligence": "document_intelligence",
    "extraction": "extraction",
    "ontology": "ontology",
    "redaction": "redaction",
    "compression": "compression",
    "enrichment": "enrichment",
    "embedding": "embedding",
    "colbert": "colbert",
    "colpali": "colpali",
}

# =============================================================================
# Service Dependency DAG (mirrors pipeline worker config.SERVICE_PARENT)
# =============================================================================

#: Each service reads its input from exactly one parent.  ``None`` means the
#: service reads from uploaded source files (always the root).
#:
#: The pipeline worker topologically sorts this DAG and groups independent
#: siblings into parallel execution waves.  For example, extraction,
#: redaction, and compression all depend only on document_intelligence and
#: can run concurrently.
SERVICE_PARENT: dict[str, str | None] = {
    "document_intelligence": None,
    "extraction": "document_intelligence",
    "redaction": "document_intelligence",
    "compression": "document_intelligence",
    "ontology": "extraction",
    "embedding": "document_intelligence",
    "colbert": "document_intelligence",
    "colpali": "document_intelligence",
}

#: Display-preference order used when the user doesn't specify ordering.
#: The worker ignores this -- it always topologically sorts.
STEP_ORDER: list[str] = [
    "document_intelligence",
    "extraction",
    "redaction",
    "ontology",
    "compression",
    "embedding",
    "colbert",
    "colpali",
    "enrichment",
    "graph_ontology_builder",
]

#: Steps that are defined but not yet implemented.
PLACEHOLDER_STEPS: dict[str, str] = {
    "enrichment": (
        "Feature Enrichment (10-dimensional per-chunk and corpus-level feature "
        "computation for retrieval-optimized data) is coming soon. Corpus-level "
        "processing requires a dedicated architecture. "
        "Follow our roadmap at https://latence.ai for updates."
    ),
    "graph_ontology_builder": (
        "Graph and Ontology Builder (cross-document entity resolution, "
        "ontology construction, and knowledge base linking) is coming soon. "
        "Follow our roadmap at https://latence.ai for updates."
    ),
}

# =============================================================================
# Smart Defaults
# =============================================================================

#: Default pipeline when the user provides only files with no step configuration.
#: Runs: OCR -> Entity Extraction -> Knowledge Graph
DEFAULT_INTELLIGENCE_PIPELINE: list[ServiceConfig] = [
    ServiceConfig(service="document_intelligence", config={"mode": "default"}),
    ServiceConfig(service="extraction", config={"label_mode": "generated"}),
    ServiceConfig(service="ontology", config={"resolve_entities": True}),
]

# =============================================================================
# Parsing Functions
# =============================================================================


def resolve_step_name(name: str) -> str:
    """Resolve a step alias to its canonical service name.

    Args:
        name: User-provided step name (e.g. ``"ocr"``, ``"knowledge_graph"``).

    Returns:
        Canonical service name (e.g. ``"document_intelligence"``).

    Raises:
        NotImplementedError: If the step is a placeholder (coming soon).
        ValueError: If the step name is not recognized.
    """
    lowered = name.lower().strip()

    # Chunking aliases: standalone only, not a pipeline step
    if lowered in ("chunking", "chunk", "split", "text_chunking"):
        raise NotImplementedError(
            "Chunking is not available as a pipeline step. "
            "Use client.chunking.chunk() for standalone text chunking. "
            "Pipeline-integrated chunking is coming in a future release."
        )

    # Check placeholders first
    if lowered in PLACEHOLDER_STEPS:
        raise NotImplementedError(PLACEHOLDER_STEPS[lowered])

    resolved = STEP_ALIASES.get(lowered)
    if resolved is None:
        known = sorted(set(STEP_ALIASES.keys()) | set(PLACEHOLDER_STEPS.keys()))
        raise ValueError(f"Unknown pipeline step '{name}'. Available steps: {', '.join(known)}")

    # Also check if the resolved name is a placeholder
    if resolved in PLACEHOLDER_STEPS:
        raise NotImplementedError(PLACEHOLDER_STEPS[resolved])

    return resolved


def _topological_sort(services: list[str]) -> list[str]:
    """Sort services respecting the ``SERVICE_PARENT`` DAG.

    Uses Kahn's algorithm.  Services not in the DAG are appended at the
    end in their original relative order.
    """
    known = [s for s in services if s in SERVICE_PARENT]
    unknown = [s for s in services if s not in SERVICE_PARENT]

    # Build adjacency from SERVICE_PARENT restricted to *requested* services
    in_degree: dict[str, int] = {s: 0 for s in known}
    children: dict[str, list[str]] = {s: [] for s in known}
    for s in known:
        parent = SERVICE_PARENT.get(s)
        if parent is not None and parent in in_degree:
            in_degree[s] += 1
            children[parent].append(s)

    queue = [s for s in known if in_degree[s] == 0]
    ordered: list[str] = []
    while queue:
        # Stable sort: pick in STEP_ORDER preference
        queue.sort(key=lambda s: STEP_ORDER.index(s) if s in STEP_ORDER else len(STEP_ORDER))
        node = queue.pop(0)
        ordered.append(node)
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(ordered) != len(known):
        cycle_nodes = set(known) - set(ordered)
        raise ValueError(f"Cycle detected in SERVICE_PARENT involving: {cycle_nodes}")

    return ordered + unknown


def parse_steps_config(steps: dict[str, dict[str, Any]]) -> list[ServiceConfig]:
    """Parse a dict-based step configuration into ordered ServiceConfig list.

    Steps are resolved through aliases and topologically sorted according
    to the ``SERVICE_PARENT`` dependency DAG (mirroring the pipeline worker).

    Args:
        steps: Mapping of step name -> config dict.
            Example: ``{"ocr": {"mode": "performance"}, "extraction": {"threshold": 0.3}}``

    Returns:
        Ordered list of ServiceConfig.

    Raises:
        NotImplementedError: If a placeholder step is requested.
        ValueError: If an unknown step name is used.
    """
    configs: dict[str, ServiceConfig] = {}

    for name, config in steps.items():
        service_name = resolve_step_name(name)
        configs[service_name] = ServiceConfig(service=service_name, config=config or {})

    ordered = _topological_sort(list(configs.keys()))
    return [configs[s] for s in ordered if s in configs]


def parse_input(
    *,
    files: list[str | Path | BinaryIO] | str | Path | None = None,
    file_urls: list[str] | None = None,
    text: str | None = None,
    entities: list[dict[str, Any]] | None = None,
) -> PipelineInput | None:
    """Parse various input formats into a PipelineInput.

    Supports:
    - Local file paths (str, Path, or BinaryIO)
    - A single file path as a string
    - Remote file URLs
    - Raw text
    - Pre-extracted entities

    Auto-detects whether a string is a file path or text based on
    extension and path-like patterns.

    Args:
        files: Local files to process.
        file_urls: URLs of remote files.
        text: Text input.
        entities: Pre-extracted entities for ontology-only pipelines.

    Returns:
        PipelineInput or None if no input provided.

    Raises:
        NotImplementedError: If S3 source input is detected.
    """
    file_inputs: list[FileInput] = []

    # Normalize single file to list
    if files is not None and not isinstance(files, list):
        files = [files]

    if files:
        for f in files:
            if isinstance(f, (str, Path)):
                f_str = str(f)
                # Check for S3 sources
                if f_str.startswith("s3://"):
                    raise NotImplementedError(
                        "S3 source input is coming soon. "
                        "For now, download the file locally or "
                        "provide a presigned URL via file_urls."
                    )
                if f_str.startswith(("http://", "https://")):
                    file_inputs.append(FileInput(url=f_str))
                else:
                    base64_data, filename = file_to_base64(f)
                    file_inputs.append(FileInput(base64=base64_data, filename=filename))
            else:
                # BinaryIO
                base64_data, filename = file_to_base64(f)
                file_inputs.append(FileInput(base64=base64_data, filename=filename))

    if file_urls:
        for url in file_urls:
            file_inputs.append(FileInput(url=url))

    if not file_inputs and not text and not entities:
        return None

    return PipelineInput(
        files=file_inputs if file_inputs else None,
        text=text,
        entities=entities,  # type: ignore[arg-type]
    )


def has_file_input(
    *,
    files: list[str | Path | BinaryIO] | str | Path | None = None,
    file_urls: list[str] | None = None,
) -> bool:
    """Check if the input includes file-based sources."""
    if files is not None:
        if isinstance(files, list):
            return len(files) > 0
        return True
    if file_urls:
        return len(file_urls) > 0
    return False


def build_pipeline_config(
    *,
    steps: dict[str, dict[str, Any]] | None = None,
    name: str | None = None,
    has_files: bool = False,
) -> PipelineConfig:
    """Build a PipelineConfig from user input.

    If no steps are provided and the input contains files, applies the
    smart default intelligence pipeline (OCR -> Extraction -> Knowledge Graph).

    Args:
        steps: Optional step configuration dict.
        name: Optional pipeline name.
        has_files: Whether the input includes files.

    Returns:
        Configured PipelineConfig.
    """
    if steps is not None:
        services = parse_steps_config(steps)
    elif has_files:
        # Smart defaults: OCR -> Extraction -> Knowledge Graph
        services = list(DEFAULT_INTELLIGENCE_PIPELINE)
    else:
        raise ValueError(
            "Either 'steps' must be provided or file input must be given "
            "(for smart defaults to apply)."
        )

    return PipelineConfig(
        services=services,
        store_intermediate=True,  # Always store intermediate for DataPackage composition
        name=name,
    )
