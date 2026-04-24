# Trace — Groundedness & Phantom-Hallucination Scoring

Score LLM outputs for groundedness against retrieval context (RAG lane),
phantom hallucinations in agentic code turns (code lane), or aggregate N
per-turn outputs into a session scoreboard (rollup lane).

The three lanes share one RunPod pod; the gateway pins the lane server-side
by URL so you cannot cross-wire a code payload into the RAG endpoint.

```python
from latence import Latence
client = Latence(api_key="lat_xxx")

# RAG groundedness
r = client.experimental.trace.rag(
    response_text="Paris is the capital of France.",
    raw_context="France's capital city is Paris.",
)
print(r.score, r.band, r.context_coverage_ratio)

# Agentic-code phantom scoring
t = client.experimental.trace.code(
    response_text="def add(a, b): return a + b",
    raw_context="# utils.py\ndef sub(a, b): return a - b",
    response_language_hint="python",
)
print(t.band, t.code_lane, t.session_signals.recommendation)

# Stateless session rollup (CPU-only, sub-ms on the pod)
rollup = client.experimental.trace.rollup(turns=[t, t])
print(rollup.noise_pct, rollup.recommendations)
```

> **Note:** Direct service APIs live under `client.experimental.*`. For
> production pipelines that orchestrate multiple services, use
> [`client.pipeline`](pipelines.md).

## Which lane should I use?

- **`rag()`** &mdash; you have a retrieval-augmented answer (response + the
  context you retrieved) and want to know whether the model actually used
  the context or hallucinated. Returns a single score, a risk band, and
  per-context-chunk coverage.
- **`code()`** &mdash; your agent is generating or editing code across
  multiple turns. Detects phantom APIs (functions/classes that don't exist
  in the provided context) and drift over turns via the opaque
  `next_session_state` handoff.
- **`rollup()`** &mdash; you already have N per-turn outputs from `rag()`
  or `code()` and want a single session scoreboard (`noise_pct`,
  `retrieval_waste_pct`, reason-code histogram, risk trail). Stateless,
  sub-ms, safe on every keystroke.

## Methods

### `client.experimental.trace.rag()`

Score a response for groundedness against retrieval context.

At least **one** of `raw_context`, `chunk_ids`, or `support_units` must be
supplied — the SDK enforces this client-side before the HTTP round-trip.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `response_text` | `str` | **required** | Generated response text to score. |
| `query_text` | `str \| None` | `None` | Optional query for query-conditioned diagnostics. |
| `raw_context` | `str \| None` | `None` | Raw context string to segment and encode on demand. |
| `chunk_ids` | `list[str \| int] \| None` | `None` | External chunk ids whose stored support vectors to reuse (fast path). |
| `support_units` | `list[SupportUnitInput \| dict] \| None` | `None` | Structured premise lane (mutually exclusive with `chunk_ids` / `raw_context`). |
| `primary_metric` | `"reverse_context" \| "triangular"` | `"reverse_context"` (server default) | Headline scalar metric. |
| `evidence_limit` | `int` | `8` (server default) | Maximum top evidence links in the sparse response (1–128). |
| `coverage_threshold` | `float` | `0.5` (server default) | Per-unit reverse-context threshold (0.0–1.0). |
| `segmentation_mode` | `"sentence" \| "sentence_packed" \| "paragraph"` | `"sentence_packed"` (server default) | How `raw_context` is segmented. |
| `attribution_mode` | `"closed_book" \| "open_domain"` | `"closed_book"` (server default) | Evidence policy. |
| `include_triangular_diagnostics` | `bool` | `True` (server default) | Include query-conditioned diagnostics. |
| `heatmap_format` | `"none" \| "data" \| "html"` | `"data"` | Heatmap surface. `html` also returns a self-contained `<div>`. |
| `verification_samples` | `list[str] \| None` | `None` | Alternate responses for semantic-entropy fusion. |
| `content_type` | `str \| None` | `None` | Structured-source hint (e.g. `application/json`). |
| `risk_band_stratum` | `str \| None` | `None` | Failure-mode hint for the calibrated risk-band classifier. |
| `model` | `str \| None` | `None` | Optional encoder override. |
| `session_id` | `str \| None` | `None` | Opaque, hashed session identifier (do NOT put user text here). |
| `request_id` | `str \| None` | `None` | Optional tracking ID. |
| `verbose` | `bool` | `False` | Return full diagnostics instead of the compact summary. |
| `return_job` | `bool` | `False` | Return `JobSubmittedResponse` for async polling. |

**Pricing:** $0.008 per request, quantized per 32,000 context tokens
(a 64k-token context counts as 2 requests).

**Client-side validation (before HTTP):**

- `ValueError("`response_text` must be a non-empty string.")` &mdash; raised when `response_text` is empty or whitespace-only.
- `ValueError("Trace scoring requires at least one of: raw_context, chunk_ids, or support_units.")` &mdash; raised when no premise lane is supplied.

### `client.experimental.trace.code()`

Superset of `rag()`, plus three code-lane-only fields:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `response_language_hint` | `"python" \| "py" \| "typescript" \| "ts" \| "tsx" \| "javascript" \| "js" \| "jsx" \| "go" \| "golang" \| "rust" \| "rs" \| None` | `None` | Hint for the AST extractor. |
| `emit_chunk_ownership` | `bool` | `False` (server default) | Return per-unit ownership table (adds a few KB over the wire). |
| `session_state` | `SessionState \| dict \| None` | `None` | Echo the previous turn's `next_session_state` verbatim to chain turns. |

**Pricing:** $2.00 per 1,000,000 aggregate tokens (counted from
`response_text` + `raw_context` + `query_text` + `support_units` with
`tiktoken`).

**Client-side validation (before HTTP):** same as `rag()` above &mdash; `response_text` must be non-empty and at least one premise lane (`raw_context`, `chunk_ids`, or `support_units`) must be supplied.

### `client.experimental.trace.rollup()`

Aggregate N per-turn outputs into a session scoreboard. Stateless,
CPU-only, sub-ms on the pod. Safe to call on every keystroke.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `turns` | `list[TraceResponse \| dict]` | **required** | Ordered per-turn outputs (response objects from `.rag()` / `.code()` work directly). |
| `session_id` | `str \| None` | `None` | Echoed on the response. |
| `heatmap_format` | `"none" \| "data" \| "html"` | `"data"` (server default) | Session-level heatmap surface. |
| `request_id` | `str \| None` | `None` | Optional tracking ID. |

**Pricing:** $0.001 flat per request.

**Client-side validation (before HTTP):**

- `ValueError("`turns` must be a non-empty list.")` &mdash; raised when `turns` is empty.

## Responses

### `TraceRagResponse` / `TraceCodeResponse`

Shared fields (both lanes):

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float \| None` | Top-level groundedness score (0–1). |
| `primary_metric` | `str \| None` | Metric backing `score`. Typically `"reverse_context"` or `"triangular"`, but the pod may echo a derived name (e.g. `"groundedness_v2"`); do not pin this to an enum on the response side. |
| `band` | `str \| None` | Risk band (`green` / `amber` / `red` / `unknown`). |
| `structured_score` | `dict \| None` | Per-component score breakdown. |
| `nli_aggregate` | `float \| None` | Aggregate NLI entailment score. |
| `context_coverage_ratio` | `float \| None` | Fraction of the response grounded in context. |
| `context_usage_ratio` | `float \| None` | Fraction of context actually used. |
| `context_unused_ratio` | `float \| None` | Fraction of context left unused. |
| `context_uncertain_ratio` | `float \| None` | Fraction with uncertain grounding. |
| `support_units_usage` | `SupportUnitsUsageSummary \| None` | Aggregate per-state counts. |
| `support_units` | `list[SupportUnitUsage] \| None` | Per-unit verdicts. |
| `latency_ms` | `float \| None` | Pod-side scoring latency. |
| `scoring_mode` | `"rag" \| "code" \| None` | Lane echoed back by the pod. |
| `session_id` | `str \| None` | Echoed session id. |
| `version` | `str \| None` | Pod handler version tag. |
| `reason` | `str \| None` | Human-readable reason for the band. |
| `warnings` | `list[str] \| None` | Non-fatal warnings. |
| `file_attribution` | `FileAttribution \| None` | Per-file owner share + reason codes. |
| `heatmap` | `Heatmap \| None` | Structured heatmap payload. |
| `heatmap_html` | `str \| None` | Self-contained `<div>` (only when `heatmap_format="html"`). |

Code-lane-only additions on `TraceCodeResponse`:

| Field | Type | Description |
|-------|------|-------------|
| `code_lane` | `CodeLaneDiagnostics \| None` | Composite / AST / NLI diagnostics for the turn. |
| `next_session_state` | `SessionState \| None` | Opaque state to pass into the next `code()` call. |
| `session_signals` | `SessionSignals \| None` | EMA groundedness, drift, phantom rate, `recommendation`. |

### `TraceRollupResponse`

| Field | Type | Description |
|-------|------|-------------|
| `turns_processed` | `int \| None` | Number of per-turn outputs aggregated. |
| `noise_pct` | `float \| None` | Fraction of turns flagged as noise. |
| `model_drift_pct` | `float \| None` | Fraction of turns with model drift. |
| `retrieval_waste_pct` | `float \| None` | Fraction of retrieved context left unused. |
| `reason_code_histogram` | `dict[str, int] \| None` | Count of each reason code over the window. |
| `recommendations` | `list[str] \| None` | Session-level recommendations. |
| `risk_band_trail` | `list[str] \| None` | Risk band per turn, chronological. |
| `drift_trend` | `dict[str, Any] \| None` | Drift trajectory summary (`{"last", "max", "mean", "min", ...}`). |
| `top_dead_files` | `list \| None` | Files consistently marked dead-weight. |
| `heatmap` / `heatmap_html` | — | Session-level heatmap payloads. |
| `session_id` | `str \| None` | Echoed session id. |

## Multi-turn session chaining (code lane)

Pass the previous response's `next_session_state` into the next call via
`session_state=prev.next_session_state`. The payload is fully opaque to
the client — just round-trip it.

```python
trace = client.experimental.trace

turn1 = trace.code(
    response_text="def add(a, b): return a + b",
    raw_context="# utils.py\ndef sub(a, b): return a - b",
    response_language_hint="python",
)

turn2 = trace.code(
    response_text="def mul(a, b): return a * b",
    raw_context="# utils.py\ndef sub(a, b): return a - b",
    response_language_hint="python",
    session_state=turn1.next_session_state,  # <- chain
)

print(turn2.session_signals.ema_groundedness)
print(turn2.session_signals.recommendation)  # "continue" / "re_anchor" / "fresh_chat"
```

## Live session scoreboard (rollup)

```python
turns = [turn1, turn2, turn3]  # mix TraceRagResponse / TraceCodeResponse / dicts
rollup = client.experimental.trace.rollup(turns=turns, session_id="sess_abc")

print(f"noise:     {rollup.noise_pct:.0%}")
print(f"drift:     {rollup.model_drift_pct:.0%}")
print(f"waste:     {rollup.retrieval_waste_pct:.0%}")
print(f"top reason codes: {rollup.reason_code_histogram}")
print(f"risk trail: {rollup.risk_band_trail}")
print(f"recommend:  {rollup.recommendations}")
```

## Using `SupportUnitInput`

Structured premises with per-unit provenance (speaker, timestamp, source id)
that the scorer propagates into `support_units_usage` on the response.

```python
from latence import SupportUnitInput

units = [
    SupportUnitInput(text="Paris is the capital of France.", source_id="doc-42"),
    SupportUnitInput(text="It is located on the Seine.",    source_id="doc-42"),
    {"text": "Population: 2.1M.", "source_id": "wiki"},
]

r = client.experimental.trace.rag(
    response_text="Paris, France's capital, sits on the Seine.",
    support_units=units,
)

for unit in (r.support_units or []):
    print(unit.source_id, unit.usage_state, unit.coverage_score)
```

## Async

```python
from latence import AsyncLatence

async with AsyncLatence(api_key="lat_xxx") as client:
    r = await client.experimental.trace.rag(
        response_text="Paris is the capital of France.",
        raw_context="France's capital city is Paris.",
    )
```

## Background jobs

```python
job = client.experimental.trace.rag(
    response_text="Paris is the capital of France.",
    raw_context="France's capital city is Paris.",
    return_job=True,
)
print(job.job_id)
result = client.jobs.wait(job.job_id)
```
