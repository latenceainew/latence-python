"""Pipeline module for the Latence SDK.

Contains the pipeline builder, validator, job handles, data package,
and specification utilities.
"""

from __future__ import annotations

from .builder import PipelineBuilder
from .config_loader import PipelineConfigError, load_pipeline_config
from .data_package import DataPackage
from .job import AsyncJob, Job
from .spec import (
    DEFAULT_INTELLIGENCE_PIPELINE,
    SERVICE_PARENT,
    STEP_ALIASES,
    STEP_ORDER,
    build_pipeline_config,
    has_file_input,
    parse_input,
    parse_steps_config,
    resolve_step_name,
)
from .validator import (
    SERVICE_IO,
    PipelineValidationError,
    validate_pipeline,
)

__all__ = [
    # Builder
    "PipelineBuilder",
    # Job handles
    "Job",
    "AsyncJob",
    # Data package
    "DataPackage",
    # Spec & config
    "SERVICE_PARENT",
    "STEP_ALIASES",
    "STEP_ORDER",
    "DEFAULT_INTELLIGENCE_PIPELINE",
    "build_pipeline_config",
    "has_file_input",
    "parse_input",
    "parse_steps_config",
    "resolve_step_name",
    # Config loader
    "PipelineConfigError",
    "load_pipeline_config",
    # Validator
    "PipelineValidationError",
    "SERVICE_IO",
    "validate_pipeline",
]
