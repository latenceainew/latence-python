# Pipeline Module Specification

## Overview

The pipeline module will provide predefined best-practice pipelines that chain Latence services together. This enables users to accomplish complex document processing workflows with minimal code while ensuring correct service ordering and data flow.

## Design Principles

1. **Predefined pipelines only** - We offer curated, tested pipelines rather than arbitrary service chaining
2. **Validation at construction** - Pipelines validate input compatibility before execution
3. **Transparent execution** - Each step's output is accessible for debugging
4. **Consistent interface** - All pipelines share a common API

## Planned Pipelines

### 1. DocumentAnalysisPipeline

**Flow:** Document Intelligence → Entity Extraction → Relation Extraction

```python
from latence import Latence
from latence.pipelines import DocumentAnalysisPipeline

client = Latence(api_key="lat_xxx")
pipeline = DocumentAnalysisPipeline(client)

result = pipeline.run(
    file_url="https://example.com/doc.pdf",
    entity_labels=["person", "organization", "location"],
    build_knowledge_graph=True
)

# Access results from each stage
print(result.document.content)
print(result.entities)
print(result.knowledge_graph)
```

### 2. SecureDocumentPipeline

**Flow:** Document Intelligence → Redaction → Extraction

For processing documents that may contain PII before entity extraction.

```python
pipeline = SecureDocumentPipeline(client)

result = pipeline.run(
    file_url="https://example.com/doc.pdf",
    redaction_mode="mask",
    entity_labels=["organization", "date", "amount"]
)

print(result.redacted_text)
print(result.entities)  # Extracted from redacted text
```

### 3. SearchIndexPipeline

**Flow:** Document Intelligence → Embedding (+ optional ColBERT/ColPali)

For preparing documents for vector search.

```python
pipeline = SearchIndexPipeline(client)

result = pipeline.run(
    file_url="https://example.com/doc.pdf",
    embedding_dimension=512,
    include_colbert=True
)

print(result.content)
print(result.dense_embedding)
print(result.colbert_embedding)
```

### 4. PromptCompressionPipeline

**Flow:** Compression → (pass to LLM)

For compressing long contexts before sending to LLMs.

```python
pipeline = PromptCompressionPipeline(client)

compressed = pipeline.run(
    messages=[
        {"role": "user", "content": "...long context..."},
        {"role": "assistant", "content": "...response..."},
    ],
    target_compression=0.5
)

print(compressed.messages)  # Ready for LLM
print(compressed.tokens_saved)
```

## Pipeline Result Structure

All pipeline results inherit from a common base:

```python
class PipelineResult(BaseModel):
    success: bool
    stages_completed: list[str]
    total_credits_used: float
    execution_time_ms: float
    
    # Stage-specific results as optional fields
    document: DocumentResult | None = None
    entities: list[Entity] | None = None
    knowledge_graph: KnowledgeGraph | None = None
    embeddings: list[list[float]] | None = None
    redacted_text: str | None = None
```

## Error Handling

Pipelines will:
1. Validate inputs before starting
2. Stop on first failure
3. Return partial results with error information
4. Provide clear error messages indicating which stage failed

```python
try:
    result = pipeline.run(file_url="...")
except PipelineError as e:
    print(f"Failed at stage: {e.failed_stage}")
    print(f"Completed stages: {e.completed_stages}")
    print(f"Partial results: {e.partial_results}")
```

## Configuration

Pipelines accept configuration for each stage:

```python
pipeline = DocumentAnalysisPipeline(
    client,
    document_config={"mode": "performant"},
    extraction_config={"label_mode": "hybrid", "enable_refinement": True},
    ontology_config={"resolve_entities": True}
)
```

## Background Execution

Pipelines support background execution for long documents:

```python
job = pipeline.run(file_url="...", return_job=True)
print(job.job_id)

# Poll for completion
result = pipeline.wait(job.job_id)
```

## Implementation Timeline

1. **Phase 1:** Core pipeline infrastructure (base classes, validation)
2. **Phase 2:** DocumentAnalysisPipeline
3. **Phase 3:** SecureDocumentPipeline, SearchIndexPipeline
4. **Phase 4:** PromptCompressionPipeline

## Notes

- Pipelines are sugar on top of the core SDK - users can always chain services manually
- We prioritize correctness and clarity over maximum flexibility
- Each pipeline is thoroughly tested end-to-end before release
