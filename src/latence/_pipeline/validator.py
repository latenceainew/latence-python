"""Pipeline validation logic for service chains.

Validates pipelines against the worker's DAG execution model
(``SERVICE_PARENT``).  The worker topologically sorts services and
runs independent branches in parallel -- validation mirrors that.

This module defines:
- SERVICE_IO: Input/output types for each service
- validate_pipeline: DAG-aware validation with optional auto-injection
"""

from __future__ import annotations

from typing import Literal

from .._exceptions import LatenceError
from .._models.pipeline import (
    PipelineConfig,
    PipelineInput,
    PipelineValidationResult,
)
from .spec import SERVICE_PARENT

# Input/output type definitions for each service (mirrors worker config.SERVICE_IO)
SERVICE_IO: dict[str, dict[str, list[str] | str]] = {
    "document_intelligence": {
        "input": ["file", "image"],
        "output": "text",
    },
    "extraction": {
        "input": ["text"],
        "output": "entities",
    },
    "ontology": {
        "input": ["text", "entities"],
        "output": "knowledge_graph",
    },
    "redaction": {
        "input": ["text"],
        "output": "text",
    },
    "compression": {
        "input": ["text"],
        "output": "text",
    },
    "embedding": {
        "input": ["text"],
        "output": "embeddings",
    },
    "colbert": {
        "input": ["text"],
        "output": "embeddings",
    },
    "colpali": {
        "input": ["image", "text"],
        "output": "embeddings",
    },
}

# Services that can produce each data type
PRODUCERS: dict[str, list[str]] = {
    "text": ["document_intelligence", "redaction", "compression"],
    "entities": ["extraction"],
    "embeddings": ["embedding", "colbert", "colpali"],
    "knowledge_graph": ["ontology"],
}

FILE_TO_TEXT_SERVICE = "document_intelligence"


class PipelineValidationError(LatenceError):
    """Raised when pipeline validation fails in strict mode."""

    def __init__(
        self,
        message: str,
        errors: list[str] | None = None,
        suggestion: str | None = None,
    ):
        self.errors = errors or []
        self.suggestion = suggestion
        super().__init__(message)


def _detect_input_type(
    pipeline_input: PipelineInput | None,
) -> Literal["file", "text", "entities", "unknown"]:
    """Detect the primary input type from pipeline input."""
    if pipeline_input is None:
        return "unknown"

    if pipeline_input.files and len(pipeline_input.files) > 0:
        return "file"
    elif pipeline_input.text:
        return "text"
    elif pipeline_input.entities and len(pipeline_input.entities) > 0:
        return "entities"

    return "unknown"


def _get_service_list(config: PipelineConfig) -> list[str]:
    """Extract service names from config."""
    return [s.service for s in config.services]


def _check_first_service_compatibility(
    input_type: str,
    first_service: str,
) -> tuple[bool, str | None]:
    """Check if the first service is compatible with the input type."""
    service_inputs = SERVICE_IO.get(first_service, {}).get("input", [])

    # Map input_type to service input types
    if input_type == "file":
        # Files need document_intelligence or colpali
        if "file" in service_inputs or "image" in service_inputs:
            return True, None
        return (
            False,
            f"Input is file/image but first service '{first_service}' requires {service_inputs}",
        )

    elif input_type == "text":
        if "text" in service_inputs:
            return True, None
        # Special case: colpali can take text for query embedding
        if first_service == "colpali" and "text" in service_inputs:
            return True, None
        return False, f"Input is text but first service '{first_service}' requires {service_inputs}"

    elif input_type == "entities":
        if "entities" in service_inputs:
            return True, None
        return (
            False,
            f"Input is entities but first service '{first_service}' requires {service_inputs}",
        )

    return True, None  # Unknown input type - let it through


def _check_dag_dependencies(
    services: list[str],
    input_type: str = "unknown",
) -> tuple[bool, list[str], list[str]]:
    """Check that every service's parent (per ``SERVICE_PARENT``) is present.

    When the input is text or entities, ``document_intelligence`` is not
    required -- text-consuming services can read the provided text directly.

    Returns:
        (all_satisfied, missing_services, errors)
    """
    errors: list[str] = []
    missing: list[str] = []

    for service in services:
        parent = SERVICE_PARENT.get(service)
        if parent is not None and parent not in services:
            if parent == "document_intelligence" and input_type in ("text", "entities"):
                continue
            missing.append(parent)
            errors.append(
                f"Service '{service}' requires its parent '{parent}' to be in the pipeline"
            )

    return len(errors) == 0, list(set(missing)), errors


def _check_service_chain_compatibility(
    services: list[str],
) -> tuple[bool, list[str]]:
    """DAG-aware compatibility: verify each service's required data types
    can be produced by its ancestors in the DAG.
    """
    errors: list[str] = []

    available_data: set[str] = set()

    for service in services:
        service_inputs = SERVICE_IO.get(service, {}).get("input", [])
        service_output = SERVICE_IO.get(service, {}).get("output", "unknown")

        parent = SERVICE_PARENT.get(service)
        if parent is not None and parent in [s for s in services]:
            if service == "ontology":
                if "text" not in available_data:
                    errors.append("Service 'ontology' requires 'text' but it's not available")
                if "entities" not in available_data and "extraction" not in services:
                    errors.append("Service 'ontology' requires 'extraction' to provide entities")
            else:
                if not any(inp in available_data for inp in service_inputs):
                    errors.append(
                        f"Service '{service}' requires one of {service_inputs} "
                        f"but available data types are {available_data or 'none'}"
                    )

        if service_output != "unknown":
            available_data.add(service_output)  # type: ignore[arg-type]
        if service == "document_intelligence":
            available_data.add("text")
        if service == "extraction":
            available_data.add("text")
            available_data.add("entities")

    return len(errors) == 0, errors


def _auto_inject_services(
    input_type: str,
    services: list[str],
) -> tuple[list[str], list[str]]:
    """Auto-inject missing parent services from the DAG.

    Only injects ``document_intelligence`` when the input is file-based.
    For text/entities input, text-consuming services don't need their
    DAG parent since the text is provided directly.

    Returns:
        (final_services, auto_injected)
    """
    final_services = list(services)
    auto_injected: list[str] = []

    if input_type == "file":
        if "document_intelligence" not in final_services:
            first_service = final_services[0] if final_services else None
            if first_service:
                first_inputs = SERVICE_IO.get(first_service, {}).get("input", [])
                if "file" not in first_inputs and "image" not in first_inputs:
                    final_services.insert(0, "document_intelligence")
                    auto_injected.append("document_intelligence")

    changed = True
    while changed:
        changed = False
        for service in list(final_services):
            parent = SERVICE_PARENT.get(service)
            if parent is not None and parent not in final_services:
                # Don't inject document_intelligence for text/entities input --
                # text-consuming services can read the provided text directly.
                if parent == "document_intelligence" and input_type in ("text", "entities"):
                    continue
                idx = final_services.index(service)
                final_services.insert(idx, parent)
                auto_injected.append(parent)
                changed = True

    return final_services, auto_injected


def validate_pipeline(
    config: PipelineConfig,
    pipeline_input: PipelineInput | None = None,
) -> PipelineValidationResult:
    """Validate a pipeline configuration.

    Args:
        config: The pipeline configuration to validate
        pipeline_input: Optional input to validate against

    Returns:
        PipelineValidationResult with validation details

    Raises:
        PipelineValidationError: If strict_mode=True and validation fails
    """
    errors: list[str] = []
    warnings: list[str] = []
    auto_injected: list[str] = []

    services = _get_service_list(config)

    # Check for empty pipeline
    if not services:
        errors.append("Pipeline must contain at least one service")
        if config.strict_mode:
            raise PipelineValidationError(
                message="Pipeline validation failed",
                errors=errors,
                suggestion="Add at least one service to the pipeline",
            )
        return PipelineValidationResult(
            valid=False,
            services=services,
            auto_injected=[],
            errors=errors,
            warnings=warnings,
        )

    # Detect input type
    input_type = _detect_input_type(pipeline_input)

    if input_type == "unknown" and pipeline_input is not None:
        warnings.append("Could not determine input type - validation may be incomplete")

    # If not strict mode, try to auto-inject missing services
    if not config.strict_mode:
        services, auto_injected = _auto_inject_services(input_type, services)
        if auto_injected:
            warnings.append(f"Auto-injected services: {auto_injected}")

    # 1. Check first service compatibility with input
    if input_type != "unknown":
        compatible, error = _check_first_service_compatibility(input_type, services[0])
        if not compatible and error:
            errors.append(error)

    # 2. Check DAG parent dependencies
    deps_satisfied, missing, dep_errors = _check_dag_dependencies(services, input_type)
    errors.extend(dep_errors)

    # 3. Check data-type chain compatibility
    chain_valid, chain_errors = _check_service_chain_compatibility(services)
    errors.extend(chain_errors)

    # ColPali with text-only input
    if "colpali" in services and input_type == "text":
        if len(services) == 1:
            warnings.append(
                "ColPali with text input is typically used for queries. "
                "For text embeddings, consider 'embedding' or 'colbert' instead."
            )

    # Raise error in strict mode if validation failed
    if errors and config.strict_mode:
        suggestion = None
        if missing:
            suggestion = f"Add the following services: {missing}"
        raise PipelineValidationError(
            message="Pipeline validation failed",
            errors=errors,
            suggestion=suggestion,
        )

    return PipelineValidationResult(
        valid=len(errors) == 0,
        services=services,
        auto_injected=auto_injected,
        errors=errors,
        warnings=warnings,
    )
