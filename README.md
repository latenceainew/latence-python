<p align="center">
  <img src="https://www.latence.ai/icon.svg" alt="Latence" width="120">
</p>

<h1 align="center">Latence Python SDK</h1>

<p align="center">
  <strong>Catch hallucinations, drift, and unused context before your users do.</strong><br>
  Groundedness scoring for RAG pipelines and AI coding agents, with a one-call path to upgrade data quality &mdash; from messy input files to fully generated markdown and knowledge graphs &mdash; as well as a high-performance retrieval engine (OSS).
</p>

<p align="center">
  <em>Charge your RAG pipelines and harnesses based on real data.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/latence/"><img src="https://img.shields.io/pypi/v/latence?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/latence/"><img src="https://img.shields.io/pypi/pyversions/latence" alt="Python"></a>
  <a href="https://github.com/latenceai/latence-python/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#step-1--trace-your-answers">Trace</a> &bull;
  <a href="#step-2--upgrade-data-quality">Upgrade Data Quality</a> &bull;
  <a href="#step-3--upgrade-retrieval">Upgrade Retrieval</a> &bull;
  <a href="docs/trace.md">Trace Reference</a> &bull;
  <a href="SDK_TUTORIAL.md">Full Tutorial</a>
</p>

---

## Quickstart

```bash
pip install latence
export LATENCE_API_KEY="lat_..."
```

```python
from latence import Latence

client = Latence()  # reads LATENCE_API_KEY from the environment

r = client.experimental.trace.rag(
    response_text="Paris is the capital of France.",
    raw_context="France's capital city is Paris.",
)
print(r.score, r.band, r.context_coverage_ratio, r.context_unused_ratio)
```

That's it. You now know whether the answer was grounded, how much of your retrieved context was actually used, and whether to trust it.

---

## Step 1 &mdash; Trace your answers

Three lanes, one mental model. Pick the one that matches what your app is doing right now.

### RAG groundedness &mdash; did the answer actually come from your context?

```python
from latence import Latence

client = Latence()

r = client.experimental.trace.rag(
    response_text="Paris is the capital of France.",
    raw_context="France's capital city is Paris.",
)

print(r.score)                   # 0.0 - 1.0
print(r.band)                    # "green" | "amber" | "red" | "unknown"
print(r.context_coverage_ratio)  # how much of the answer is grounded in context
print(r.context_unused_ratio)    # how much retrieved context was dead weight
```

### Code agents &mdash; catch phantom APIs and drift turn-over-turn

Chain turns with the opaque `next_session_state` handoff. The SDK never forces you to track session internals.

```python
turn1 = client.experimental.trace.code(
    response_text="def add(a, b): return a + b",
    raw_context="# utils.py\ndef sub(a, b): return a - b",
    response_language_hint="python",
)

turn2 = client.experimental.trace.code(
    response_text="def mul(a, b): return a * b",
    raw_context="# utils.py\ndef sub(a, b): return a - b",
    response_language_hint="python",
    session_state=turn1.next_session_state,   # chain turns
)

print(turn2.band)
print(turn2.session_signals.recommendation)   # "continue" | "re_anchor" | "fresh_chat"
```

Hosted Trace pricing is $0.008/request by default. For higher-cost quality
mode, pass `profile="quality"` to `trace.rag(...)` or `trace.code(...)`;
quality requests bill at $0.016/request.

### Session rollup &mdash; one scoreboard for a live session

Stateless, CPU-only, sub-ms on the pod. Safe to call on every keystroke.

```python
rollup = client.experimental.trace.rollup(turns=[turn1, turn2])

print(rollup.noise_pct)              # fraction of turns flagged as noise
print(rollup.retrieval_waste_pct)    # fraction of retrieved context left unused
print(rollup.model_drift_pct)        # fraction of turns with drift
print(rollup.reason_code_histogram)  # why the turns failed, aggregated
print(rollup.risk_band_trail)        # per-turn band, chronological
print(rollup.recommendations)        # actionable session-level advice
```

### What the signals tell you to do next

The numbers above are not diagnostics. They are routing rules:

| Signal | Meaning | Next step |
|---|---|---|
| `band` amber/red, low `context_coverage_ratio` | The answer isn't grounded in what you retrieved. | **[Upgrade data quality](#step-2--upgrade-data-quality)** &mdash; your upstream documents are the bottleneck. |
| High `context_unused_ratio`, `retrieval_waste_pct > 30%` | You retrieved the wrong chunks. | **[Upgrade retrieval](#step-3--upgrade-retrieval)** &mdash; your retriever is the bottleneck. |
| `session_signals.recommendation = "re_anchor"` / `"fresh_chat"` on the code lane | Session drift is compounding. | Reset the agent's context on the next turn. |

Full reference: [Trace docs](docs/trace.md) and [SDK tutorial &sect;18](SDK_TUTORIAL.md#18-direct-api-trace-groundedness--phantom-scoring).

### Async

Every method above has an `await`-able twin under `AsyncLatence`:

```python
from latence import AsyncLatence

async with AsyncLatence() as client:
    r = await client.experimental.trace.rag(
        response_text="Paris is the capital of France.",
        raw_context="France's capital city is Paris.",
    )
```

---

## Step 2 &mdash; Upgrade data quality

Trace is showing low coverage or amber/red bands? The model is rarely the problem. It's usually the upstream data: un-OCR'd PDFs, missing entities, unresolved references. The Latence **Data Intelligence Pipeline** cleans that in one call.

```python
job = client.pipeline.run(files=["contract.pdf"])
pkg = job.wait_for_completion()

print(pkg.document.markdown)                         # clean markdown
print(pkg.entities.summary)                          # {"total": 142, "by_type": {...}}
print(pkg.knowledge_graph.summary.total_relations)   # 87
pkg.download_archive("./results.zip")
```

Smart defaults: OCR &rarr; entity extraction &rarr; relation extraction. Configure any step explicitly:

```python
job = client.pipeline.run(
    files=["contract.pdf"],
    steps={
        "ocr": {"mode": "performance"},
        "redaction": {"mode": "balanced", "redact": True},
        "extraction": {"label_mode": "hybrid", "threshold": 0.3},
        "relation_extraction": {"resolve_entities": True},
    },
)
```

Every run returns a structured `DataPackage`:

- `pkg.document` &mdash; markdown + per-page layout (OCR)
- `pkg.entities` &mdash; entity list + summary (extraction)
- `pkg.knowledge_graph` &mdash; entities + relations + graph summary (relation extraction)
- `pkg.redaction` &mdash; cleaned text + PII list (redaction)
- `pkg.compression` &mdash; compressed text + ratio (compression)
- `pkg.quality` &mdash; per-stage confidence, latency, cost

Power users: the typed [`PipelineBuilder`](SDK_TUTORIAL.md#4-pipeline-fluent-builder) accepts YAML and validates client-side. See [docs/pipelines.md](docs/pipelines.md) for the full orchestration reference (DAG execution, resumable jobs, progress callbacks).

### Corpus-level: Dataset Intelligence

Feed pipeline outputs into `client.experimental.dataset_intelligence_service` to build corpus-wide knowledge graphs, ontologies, and enriched feature spaces with incremental ingestion:

| Tier | Method | What it does |
|------|--------|-------------|
| 1 | `di.enrich()` | Semantic feature vectors (CPU-only, fast) |
| 2 | `di.build_graph()` | Entity resolution, knowledge graph, link prediction |
| 3 | `di.build_ontology()` | Concept clustering, hierarchy induction |
| Full | `di.run()` | All three tiers sequentially |

See [docs/dataset_intelligence.md](docs/dataset_intelligence.md).

---

## Step 3 &mdash; Upgrade retrieval

If Trace keeps flagging a high `context_unused_ratio`, or the session rollup shows `retrieval_waste_pct > 30%`, your model isn't the problem &mdash; **your retrieval engine is shipping the wrong chunks**.

&rarr; **[ColSearch &mdash; High Performance Late Interaction and multimodal search engine](https://github.com/ddickmann/colsearch)**

ColSearch is our late-interaction retrieval engine: token-level ColBERT recall, native multimodal search over PDFs and images, and a drop-in replacement for the retrieval step in your RAG stack. Wire it in and `context_unused_ratio` collapses.

---

## Error handling

```python
from latence import (
    LatenceError, AuthenticationError, InsufficientCreditsError,
    RateLimitError, JobError, JobTimeoutError, TransportError,
)

try:
    r = client.experimental.trace.rag(
        response_text="Paris is the capital of France.",
        raw_context="France's capital city is Paris.",
    )
except AuthenticationError:
    ...  # 401
except InsufficientCreditsError:
    ...  # 402
except RateLimitError as e:
    ...  # 429, retry after e.retry_after
except JobError as e:
    ...  # pipeline job failed; check e.is_resumable
except TransportError:
    ...  # network / DNS
```

The SDK retries on 429 and 5xx with exponential backoff (default 2 retries, respects `Retry-After`).

---

## Configuration

```bash
export LATENCE_API_KEY="lat_your_key"
```

```python
from latence import Latence
import latence

client = Latence(
    api_key="lat_...",       # or LATENCE_API_KEY env var
    base_url="https://...",  # or LATENCE_BASE_URL env var
    timeout=60.0,            # request timeout (default: 60s)
    max_retries=2,           # retry attempts (default: 2)
)

latence.setup_logging("DEBUG")  # logs every HTTP request/response
```

---

## Resources

| | |
|---|---|
| **Trace reference** | [docs/trace.md](docs/trace.md) &mdash; parameters and full response schema |
| **Full tutorial** | [SDK_TUTORIAL.md](SDK_TUTORIAL.md) &mdash; every service, every parameter |
| **API docs** | [docs.latence.ai](https://docs.latence.ai) |
| **Portal** | [app.latence.ai](https://app.latence.ai) |

---

<p align="center">
  <sub>MIT License &bull; <a href="https://latence.ai">latence.ai</a></sub>
</p>
