# Enrichment

Transform raw text into retrieval-optimized chunks enriched with 10 feature dimensions. One call gives you embeddings, quality signals, structural metadata, semantic roles, and corpus-level analytics — everything you need for production RAG.

```python
from latence import Latence
client = Latence(api_key="lat_xxx")

result = client.experimental.enrichment.enrich(
    text="Your document text here...",
    strategy="hybrid",
    chunk_size=512,
)
print(f"{result.data.num_chunks} chunks, {result.data.embedding_dim}-dim embeddings")
print(f"Mean quality: {result.data.features['quality']['aggregate']['mean_coherence']:.2f}")
```

---

## Chunking

Before enrichment features can be computed, the document is split into overlapping segments. Four strategies are available — each optimizing for a different trade-off between speed, boundary quality, and semantic coherence.

### `client.experimental.enrichment.chunk()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Input text (1–5M chars) |
| `strategy` | `str` | `"hybrid"` | `character`, `token`, `semantic`, `hybrid` |
| `chunk_size` | `int` | `512` | Target chunk size |
| `chunk_overlap` | `int` | `50` | Overlap between adjacent chunks |
| `min_chunk_size` | `int` | `64` | Minimum chunk size |
| `request_id` | `str \| None` | `None` | Optional tracking ID |
| `return_job` | `bool` | `False` | Return job ID for async polling |

---

### Strategy: `character`

**What it does.** Splits text by character count, snapping boundaries to the nearest paragraph, sentence, or whitespace boundary to avoid cutting mid-word.

**How it works.** A sliding window of `chunk_size` characters advances through the text. At each step, the algorithm searches backward from the target boundary for the best semantic break point: paragraph breaks (`\n\n`), sentence-ending punctuation (`. `, `! `, `? `), newlines, or spaces — in that priority order.

**When to use.** The fastest strategy. Use when processing speed matters more than boundary precision — for example, when downstream models can tolerate slightly noisy boundaries, or when the text has minimal structure. Also ideal for very large documents (>1 MB) where the constant overhead of tokenization or embedding would be prohibitive.

**Trade-offs.** Chunk sizes are measured in characters, not tokens, so token counts may vary within a chunk. Does not consider semantic coherence.

---

### Strategy: `token`

**What it does.** Splits text so each chunk contains approximately `chunk_size` **tokens** (as counted by OpenAI's tiktoken `o200k_base` tokenizer), then snaps to semantic boundaries.

**How it works.** Uses a binary search to find the character position where the token count reaches the target. The tokenizer is initialized once as a singleton at service startup (Rust-backed, thread-safe). After finding the token boundary, the same boundary-snapping logic as CHARACTER is applied.

**When to use.** Essential when your downstream LLM has strict token limits or when billing is per-token. Guarantees each chunk fits within a known token budget.

**Trade-offs.** Slightly slower than CHARACTER because each boundary requires tokenization. Still does not consider semantic coherence between sentences.

---

### Strategy: `semantic`

**What it does.** Splits text based on **embedding similarity boundaries** — chunks are formed where the topic changes, rather than at fixed-length intervals.

**How it works.**
1. **Sentence tokenization**: The text is split into sentences using regex patterns.
2. **Sentence embedding**: All sentences are embedded in a single batch via the vLLM embedding server.
3. **Boundary detection**: A sliding window computes cosine similarity between the *preceding* and *following* window. Positions where similarity drops below the threshold are marked as topic boundaries.
4. **Chunk assembly**: Sentences between consecutive boundaries are concatenated into chunks.

Each chunk receives a `semantic_score` (0–1) measuring the average consecutive-sentence cosine similarity within that chunk — higher means more semantically cohesive.

**When to use.** Ideal for long-form documents with multiple topics (reports, textbooks, legal documents). Produces the highest-quality chunks for RAG.

**Trade-offs.** The slowest strategy because it requires embedding every sentence. Chunk sizes vary significantly.

---

### Strategy: `hybrid`

**What it does.** Combines the speed of CHARACTER splitting with the quality of SEMANTIC refinement. First pass: fixed-size character splits. Second pass: any oversized chunk (>1.5x target) is re-split using semantic boundaries.

**How it works.**
1. **Phase 1**: Run CHARACTER splitting to produce initial chunks.
2. **Phase 2**: For each chunk whose `char_count` exceeds `chunk_size × 1.5`, the chunk content is passed through the full SEMANTIC pipeline.
3. Positions of sub-chunks are adjusted back to absolute document coordinates.

**When to use.** The recommended default for production. Gives you bounded chunk sizes while still benefiting from semantic boundary detection where it matters most.

**Trade-offs.** Slightly slower than CHARACTER, but much faster than fully SEMANTIC. The best balance of speed, boundary quality, and chunk-size control.

---

### Response: `ChunkResponse`

| Field | Type | Description |
|-------|------|-------------|
| `chunks` | `list[ChunkItem]` | List of chunk objects |
| `num_chunks` | `int` | Total chunk count |
| `strategy` | `str` | Strategy used |
| `chunk_size` | `int` | Target chunk size |
| `processing_time_ms` | `float` | Server processing time |
| `usage` | `Usage \| None` | Credits consumed |

### `ChunkItem` fields

| Field | Type | Present when | Description |
|-------|------|--------------|-------------|
| `content` | `str` | always | Chunk text |
| `index` | `int` | always | 0-based position |
| `start` | `int` | always | Start char offset |
| `end` | `int` | always | End char offset |
| `char_count` | `int` | always | Character count |
| `token_count` | `int` | TOKEN strategy | Token count |
| `semantic_score` | `float` | SEMANTIC/HYBRID | Intra-chunk coherence (0–1) |
| `section_path` | `list[str]` | Markdown headings exist | Heading hierarchy |

---

## Feature Enrichment

### `client.experimental.enrichment.enrich()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Input text (1–5M chars) |
| `strategy` | `str` | `"hybrid"` | Chunking strategy |
| `chunk_size` | `int` | `512` | Target chunk size |
| `chunk_overlap` | `int` | `50` | Overlap |
| `min_chunk_size` | `int` | `64` | Min chunk size |
| `encoding_format` | `str` | `"float"` | Embedding format: `"float"` or `"base64"` |
| `features` | `list[str] \| None` | all 10 | Feature groups to compute |
| `request_id` | `str \| None` | `None` | Tracking ID |
| `return_job` | `bool` | `False` | Async polling |

**Valid feature groups:** `quality`, `density`, `structural`, `semantic`, `compression`, `zipf`, `coherence`, `spectral`, `drift`, `redundancy`

---

### Feature 1: Quality

**What it measures.** Per-chunk text quality — how well-formed and readable a chunk is.

- **Coherence score** (0–1): Ratio of meaningful words to total words, factoring in average word length and vocabulary diversity.
- **Short / Long flags**: Binary flags based on configurable thresholds (default: 20 chars short, 10,000 chars long).

**Why it matters.** Quality scoring enables *adaptive retrieval*: weight high-coherence chunks higher in search results, or filter out chunks below a threshold.

---

### Feature 2: Density

**What it measures.** Lexical richness and information density of each chunk.

- **Unique word ratio**: Fraction of distinct words over total words.
- **Technical density**: Proportion of words longer than average (correlates with domain-specific terminology).
- **Average sentence length**: Mean words per sentence.
- **Punctuation density**: Ratio of punctuation characters to total characters.

**Why it matters.** Density features enable smarter chunk prioritization: surface information-dense chunks for technical questions, de-prioritize formulaic content.

---

### Feature 3: Structural

**What it measures.** Layout and position of each chunk within the document.

- **Heading count**, **List count**, **Code block count**, **Link count**, **Cross-reference count**
- **Relative position** (0–1): Chunk's position in the document.
- **Recency score** (optional): Temporal decay score if a reference date is configured.

**Why it matters.** Structural metadata enables *structure-aware retrieval*: boost chunks with headings, filter to specific document sections, implement positional biases.

---

### Feature 4: Semantic

**What it measures.** Rhetorical role and centrality of each chunk.

- **Rhetorical role**: Zero-shot classification into 9 roles (definition, evidence, claim, procedure, example, conclusion, context, data, unknown).
- **Centrality** (0–1): Cosine similarity between the chunk embedding and the document-level embedding.

**Why it matters.** Rhetorical roles unlock *query-type routing*: match user intent to content type. Centrality enables document summarization (top-k by centrality).

---

### Feature 5: Compression

**What it measures.** How compressible each chunk is — the ratio of unique information to total content.

- **Compression ratio**: Ratio of compressed text length (zlib) to original length.
- **Unique token ratio**: Fraction of distinct tokens over total token count.

**Why it matters.** Chunks with ratio < 0.4 are almost always repetitive artifacts and can be safely de-prioritized.

---

### Feature 6: Zipf

**What it measures.** Word frequency distribution, measuring adherence to Zipf's law.

- **Alpha**: Zipf exponent (natural English ≈ 1.0, legal German ≈ 0.3–0.5).
- **Vocab size**: Number of unique words.
- **Fit quality** (R²): How well the power-law model fits the data.

**Why it matters.** Language-independent anomaly detector. Unusual alpha values flag non-prose content (tables, code, structured data).

---

### Feature 7: Coherence (corpus-level)

**What it measures.** Semantic flow between consecutive chunks across the entire document.

- **Mean / Min / Max / Std similarity**: Pairwise cosine similarity between consecutive chunk embeddings.

**Why it matters.** High coherence with low variance = focused, well-organized document. Low coherence = compilation or poorly structured text.

---

### Feature 8: Spectral (corpus-level)

**What it measures.** Topical diversity via SVD of the chunk embedding matrix.

- **Effective rank**: Continuous measure of independent topic dimensions.
- **Rank ratio**: effective_rank / num_chunks. Near 1.0 = maximal diversity; near 0.0 = highly homogeneous.
- **Top singular values**: Dominant spectral components.

**Why it matters.** Tells you what kind of document you're dealing with before reading it. Informs retrieval strategy.

---

### Feature 9: Drift (corpus-level)

**What it measures.** Where in the document the topic changes significantly.

- **Similarities**: Cosine similarity between consecutive chunks.
- **Major breaks**: Indices where similarity drops below threshold — the semantic "chapter breaks."

**Why it matters.** Enables *dynamic document segmentation* by actual topic boundaries, especially powerful for OCR'd PDFs and transcripts.

---

### Feature 10: Redundancy (corpus-level)

**What it measures.** Near-duplicate chunks carrying essentially the same information.

- **Redundant pairs**: List of chunk index pairs exceeding the similarity threshold.
- **Redundancy rate**: Fraction of chunks involved in at least one redundant pair.

**Why it matters.** 0% redundancy confirms distinct chunks. Rate above 5% suggests adjusting chunk_size or overlap.

---

## Pricing

| Component | Cost |
|-----------|------|
| Enrichment (chunk + features) | $0.50 / 1M characters |
| Chunk only | $0.10 / 1M characters |

---

## Examples

### Chunk a PDF after OCR

```python
job = client.pipeline.run(files=["report.pdf"])
pkg = job.wait_for_completion()

chunks = client.experimental.enrichment.chunk(
    text=pkg.document.markdown,
    strategy="hybrid",
    chunk_size=512,
)
print(f"{chunks.data.num_chunks} chunks")
```

### Full enrichment with selective features

```python
result = client.experimental.enrichment.enrich(
    text="Your document text...",
    strategy="semantic",
    chunk_size=1024,
    features=["quality", "semantic", "drift", "redundancy"],
)

# Quality filter: keep only high-coherence chunks
quality = result.data.features["quality"]
good_chunks = [
    chunk for chunk, q in zip(result.data.chunks, quality["per_chunk"])
    if q["coherence_score"] > 0.5
]
print(f"{len(good_chunks)} / {result.data.num_chunks} chunks pass quality filter")

# Check for topic drift
drift = result.data.features["drift"]
print(f"{drift['num_major_breaks']} topic changes detected")
```

### Pipeline: OCR → Enrich → Knowledge Graph

```python
from latence import PipelineBuilder

config = (
    PipelineBuilder()
    .doc_intel(mode="performance")
    .enrichment(strategy="hybrid", chunk_size=512)
    .ontology(resolve_entities=True)
    .build()
)

job = client.pipeline.submit(config, files=["contract.pdf"])
pkg = job.wait_for_completion()

print(pkg.enrichment.summary.num_chunks)
print(pkg.enrichment.features["quality"]["aggregate"]["mean_coherence"])
```
