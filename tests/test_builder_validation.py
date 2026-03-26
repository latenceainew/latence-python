"""Tests for the builder-validator integration."""

import pytest

from latence._models.pipeline import PipelineConfig, ServiceConfig
from latence._pipeline.builder import PipelineBuilder
from latence._pipeline.validator import PipelineValidationError


# =============================================================================
# Smart defaults and auto-injection
# =============================================================================


def test_build_uses_smart_defaults_for_file():
    """PipelineBuilder().build() (default input_type='file') should produce a config with document_intelligence, extraction, ontology."""
    config = PipelineBuilder().build()
    services = [s.service for s in config.services]
    assert "document_intelligence" in services
    assert "extraction" in services
    assert "ontology" in services


def test_build_does_not_inject_doc_intel_for_text():
    """PipelineBuilder().extraction().build(input_type='text') should NOT have document_intelligence in services."""
    config = PipelineBuilder().extraction().build(input_type="text")
    services = [s.service for s in config.services]
    assert "document_intelligence" not in services
    assert "extraction" in services


def test_build_auto_injects_extraction_for_ontology():
    """PipelineBuilder().ontology().build() should auto-inject both document_intelligence and extraction."""
    config = PipelineBuilder().ontology().build()
    services = [s.service for s in config.services]
    assert "document_intelligence" in services
    assert "extraction" in services
    assert "ontology" in services


# =============================================================================
# Strict mode validation
# =============================================================================


def test_build_strict_mode_raises_for_invalid():
    """PipelineBuilder().extraction().strict().build() should raise PipelineValidationError because extraction requires doc_intel parent with file input."""
    with pytest.raises(PipelineValidationError):
        PipelineBuilder().extraction().strict().build()


# =============================================================================
# Duplicate service and alias resolution
# =============================================================================


def test_duplicate_service_raises():
    """Adding the same service twice should raise ValueError."""
    with pytest.raises(ValueError, match="already added"):
        PipelineBuilder().extraction().extraction()


def test_alias_resolution_in_add():
    """PipelineBuilder().add('ocr') should resolve to document_intelligence."""
    builder = PipelineBuilder().add("ocr")
    assert len(builder._services) == 1
    assert builder._services[0].service == "document_intelligence"


# =============================================================================
# from_yaml return type
# =============================================================================


def test_from_yaml_returns_builder():
    """Verify PipelineBuilder.from_yaml has the right return type annotation."""
    from typing import get_type_hints

    hints = get_type_hints(PipelineBuilder.from_yaml)
    # Return can be PipelineBuilder or forward reference string
    return_hint = hints.get("return")
    assert return_hint is PipelineBuilder or (
        isinstance(return_hint, str) and "PipelineBuilder" in return_hint
    )


# =============================================================================
# Cycle detection in topological sort
# =============================================================================


def test_cycle_detection():
    """Test that topological sort raises ValueError on cycle."""
    from latence._pipeline.spec import SERVICE_PARENT, _topological_sort

    # Normal path: no cycle
    result = _topological_sort(["document_intelligence", "extraction", "ontology"])
    assert result.index("document_intelligence") < result.index("extraction")
    assert result.index("extraction") < result.index("ontology")

    # Create a cycle by monkeypatching: extraction -> doc_intel, doc_intel -> extraction
    original = dict(SERVICE_PARENT)
    try:
        SERVICE_PARENT["document_intelligence"] = "extraction"
        SERVICE_PARENT["extraction"] = "document_intelligence"
        with pytest.raises(ValueError, match="Cycle detected"):
            _topological_sort(["document_intelligence", "extraction"])
    finally:
        SERVICE_PARENT.clear()
        SERVICE_PARENT.update(original)


# =============================================================================
# Client-side validation
# =============================================================================


def test_client_side_validation_extraction_threshold():
    """PipelineBuilder().extraction(threshold=1.5) should raise ValueError."""
    with pytest.raises(ValueError, match="between"):
        PipelineBuilder().extraction(threshold=1.5)


def test_client_side_validation_dimension():
    """PipelineBuilder().embedding(dimension=999) should raise ValueError."""
    with pytest.raises(ValueError, match="dimension must be one of"):
        PipelineBuilder().embedding(dimension=999)


# =============================================================================
# store_intermediate defaults
# =============================================================================


def test_store_intermediate_defaults_true():
    """PipelineBuilder().doc_intel().build() should have store_intermediate=True."""
    config = PipelineBuilder().doc_intel().build()
    assert config.store_intermediate is True


# =============================================================================
# Config mapping when auto-injection occurs
# =============================================================================


def test_auto_injected_services_preserve_user_configs():
    """When build() auto-injects services, user-provided configs must stay
    attached to their original service -- not shift by index."""
    config = (
        PipelineBuilder()
        .extraction(threshold=0.25, label_mode="hybrid", user_labels=["person"])
        .ontology(resolve_entities=True, kg_output_format="property_graph")
        .build(input_type="file")
    )
    svc_map = {sc.service: sc.config for sc in config.services}

    assert "document_intelligence" in svc_map, "doc_intel should be auto-injected"
    assert svc_map["extraction"]["threshold"] == 0.25
    assert svc_map["extraction"]["label_mode"] == "hybrid"
    assert svc_map["ontology"]["resolve_entities"] is True
    assert svc_map["ontology"]["kg_output_format"] == "property_graph"
    assert svc_map["document_intelligence"] == {}, "auto-injected service should get empty config"
