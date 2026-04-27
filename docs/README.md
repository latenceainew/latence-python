# Latence SDK Documentation

## Pipelines (Primary API)

The pipeline is the recommended way to use Latence. It chains multiple services into a single async job and returns a structured `DataPackage`.

| Feature | Guide |
|---------|-------|
| **Pipelines** -- Multi-service orchestration with DAG execution | [pipelines.md](pipelines.md) |
| **SDK Tutorial** -- Complete parameter reference and examples | [SDK_TUTORIAL.md](../SDK_TUTORIAL.md) |

```python
from latence import Latence

client = Latence()  # reads LATENCE_API_KEY from environment

# PDF -> structured knowledge (smart defaults: OCR -> Extraction -> Ontology)
job = client.pipeline.run(files=["contract.pdf"])
pkg = job.wait_for_completion()

print(pkg.entities.summary)
print(pkg.knowledge_graph.summary.total_relations)
pkg.download_archive("./results.zip")
```

## Direct Service APIs (Experimental)

Individual services are available under `client.experimental.*` for fine-grained control. For most use cases, prefer the pipeline.

| Service | Description | Guide |
|---------|-------------|-------|
| **Trace** | Groundedness + phantom-hallucination scoring (RAG / code / session rollup) | [trace.md](trace.md) |
| **Embed** (Unified) | Dense, token-level, and visual embeddings in one API | [embed.md](embed.md) |
| **Compression** | Text and chat message compression (up to 80%) | [compression.md](compression.md) |
| **Document Intelligence** | OCR and structure extraction from PDFs, images, Office docs | [document_intelligence.md](document_intelligence.md) |
| **Entity Extraction** | Zero-shot entity extraction | [extraction.md](extraction.md) |
| **Relation Extraction** | Relation extraction and knowledge graph construction | [ontology.md](ontology.md) |
| **Redaction** | PII detection and GDPR-compliant redaction | [redaction.md](redaction.md) |
| **Dataset Intelligence** | Corpus-level KG, ontology, incremental ingestion | [dataset_intelligence.md](dataset_intelligence.md) |

### Quick Navigation

```python
from latence import Latence
client = Latence()

# --- Pipeline (recommended) ---
client.pipeline.run(files=[...])                                # Submit + wait
client.pipeline.submit(config, files=[...])                     # Submit only

# --- Direct APIs (experimental) ---
client.experimental.embed.dense(...)                            # Dense embeddings
client.experimental.embed.late_interaction(...)                 # ColBERT embeddings
client.experimental.embed.image(...)                            # ColPali embeddings
client.experimental.compression.compress(...)                   # Text compression
client.experimental.compression.compress_messages(...)          # Chat compression
client.experimental.document_intelligence.process(...)          # OCR / document processing
client.experimental.extraction.extract(...)                     # Entity extraction
client.experimental.ontology.build_graph(...)                   # Knowledge graphs
client.experimental.redaction.detect_pii(...)                   # PII detection & redaction
client.experimental.trace.rag(...)                              # RAG groundedness scoring
client.experimental.trace.code(...)                             # Agentic-code phantom scoring
client.experimental.trace.rollup(turns=[...])                   # Session-level scoreboard

# --- Dataset Intelligence (corpus-level) ---
client.experimental.dataset_intelligence_service.run(...)        # Full DI pipeline
client.experimental.dataset_intelligence_service.build_graph(...)# Knowledge graph (tier 2)
client.experimental.dataset_intelligence_service.enrich(...)     # Feature enrichment (tier 1)
client.experimental.dataset_intelligence_service.build_ontology(...)  # Ontology (tier 3)

# --- Utilities ---
client.credits.balance()                                        # Check balance
client.jobs.wait(job_id)                                        # Background jobs
```

## Embeddings

The unified **Embed** API is the recommended interface for all embedding types:

- `client.experimental.embed.dense()` -- Standard vectors for semantic search
- `client.experimental.embed.late_interaction()` -- Token-level for precise matching (ColBERT)
- `client.experimental.embed.image()` -- Vision-language for visual documents (ColPali)

Legacy direct endpoints are still available:

- [embedding.md](embedding.md) -- Dense vectors (prefer `client.experimental.embed.dense()`)
- [colbert.md](colbert.md) -- Token-level (prefer `client.experimental.embed.late_interaction()`)
- [colpali.md](colpali.md) -- Visual (prefer `client.experimental.embed.image()`)
