"""YAML pipeline configuration loader.

Loads pipeline configurations from YAML files and converts them into
``PipelineConfig`` objects suitable for execution.

Example YAML:

.. code-block:: yaml

    steps:
      document_intelligence:
        mode: performance
        output_format: markdown
      extraction:
        label_mode: hybrid
        user_labels: [person, organization, location]
      ontology:
        resolve_entities: true
    name: my-pipeline
    store_intermediate: true
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._models.pipeline import PipelineConfig
from .spec import PLACEHOLDER_STEPS, STEP_ALIASES, parse_steps_config

# Accepted top-level keys in the YAML file
_TOP_LEVEL_KEYS = frozenset(
    {
        "steps",
        "name",
        "store_intermediate",
        "strict_mode",
    }
)

# Known parameters per service for validation warnings
KNOWN_PARAMS: dict[str, frozenset[str]] = {
    # Pipeline-available params only. Layout detection, chart recognition,
    # seal recognition, auto-rotate, and dewarp are locked to optimal defaults
    # in pipelines. For full control, use the direct API.
    "document_intelligence": frozenset(
        {
            "mode",
            "output_format",
            "use_ocr_for_image_block",
        }
    ),
    "redaction": frozenset(
        {
            "mode",
            "threshold",
            "redact",
            "redaction_mode",
            "normalize_scores",
            "chunk_size",
            "enable_refinement",
            "enforce_refinement",
            "refinement_threshold",
        }
    ),
    "extraction": frozenset(
        {
            "label_mode",
            "user_labels",
            "threshold",
            "flat_ner",
            "multi_label",
            "chunk_size",
            "enable_refinement",
            "enforce_refinement",
            "refinement_threshold",
        }
    ),
    "ontology": frozenset(
        {
            "relation_threshold",
            "symmetric",
            "generate_knowledge_graph",
            "max_relations_per_decode",
            "resolve_entities",
            "optimize_relations",
            "optimize_entity_resolution",
            "predict_missing_relations",
            "link_prediction_verify_with_ai",
            "kg_output_format",
            "namespace_uri",
        }
    ),
    "compression": frozenset(
        {
            "compression_rate",
            "force_preserve_digit",
            "force_tokens",
            "apply_toon",
            "chunk_size",
            "fallback_mode",
        }
    ),
}


class PipelineConfigError(Exception):
    """Raised when a pipeline YAML configuration is invalid."""


def _validate_yaml_structure(data: dict[str, Any]) -> list[str]:
    """Validate the top-level YAML structure and return warnings."""
    warnings: list[str] = []

    if not isinstance(data, dict):
        raise PipelineConfigError("Pipeline YAML must be a mapping (dict) at the top level.")

    unknown_keys = set(data.keys()) - _TOP_LEVEL_KEYS
    if unknown_keys:
        warnings.append(f"Unknown top-level keys ignored: {', '.join(sorted(unknown_keys))}")

    if "steps" not in data:
        raise PipelineConfigError("Pipeline YAML must contain a 'steps' key.")

    steps = data["steps"]
    if not isinstance(steps, dict) or not steps:
        raise PipelineConfigError(
            "The 'steps' key must be a non-empty mapping of step_name -> config."
        )

    # Validate each step
    all_known = set(STEP_ALIASES.keys()) | set(PLACEHOLDER_STEPS.keys())
    for step_name, step_config in steps.items():
        lowered = str(step_name).lower().strip()

        if lowered in PLACEHOLDER_STEPS:
            raise PipelineConfigError(
                f"Step '{step_name}' is not yet available: {PLACEHOLDER_STEPS[lowered]}"
            )

        if lowered not in all_known:
            raise PipelineConfigError(
                f"Unknown step '{step_name}'. Available: {', '.join(sorted(all_known))}"
            )

        if step_config is not None and not isinstance(step_config, dict):
            raise PipelineConfigError(
                f"Configuration for step '{step_name}' must be a mapping or null."
            )

        # Warn on unknown params
        if step_config and lowered in KNOWN_PARAMS:
            unknown_params = set(step_config.keys()) - KNOWN_PARAMS[lowered]
            if unknown_params:
                warnings.append(
                    f"Step '{step_name}' has unknown parameters: "
                    f"{', '.join(sorted(unknown_params))}"
                )

    return warnings


def load_pipeline_config(
    path: str | Path,
    *,
    strict: bool = False,
) -> tuple[PipelineConfig, list[str]]:
    """Load a pipeline configuration from a YAML file.

    Args:
        path: Path to the YAML file.
        strict: If True, raise on unknown parameters instead of warning.

    Returns:
        A tuple of (PipelineConfig, warnings).
        Warnings contain non-fatal issues like unknown parameters.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        PipelineConfigError: If the YAML is invalid.
        ImportError: If PyYAML is not installed.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML config loading. Install it with: pip install pyyaml"
        ) from None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise PipelineConfigError("Pipeline YAML file is empty.")

    warnings = _validate_yaml_structure(data)

    if strict and warnings:
        raise PipelineConfigError("Strict mode: " + "; ".join(warnings))

    steps_raw: dict[str, dict[str, Any]] = {}
    for step_name, step_config in data["steps"].items():
        steps_raw[str(step_name)] = step_config or {}

    services = parse_steps_config(steps_raw)

    config = PipelineConfig(
        services=services,
        store_intermediate=data.get("store_intermediate", True),
        name=data.get("name"),
        strict_mode=data.get("strict_mode", False),
    )

    return config, warnings
