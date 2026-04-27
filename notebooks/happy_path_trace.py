"""
Happy-path trace journey: score 50 RAGTruth samples via AsyncLatence.

Usage:
    LATENCE_API_KEY=lat_... python happy_path_trace.py
"""

import asyncio
import json
import os
import sys
import time
from collections import Counter

from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from latence import AsyncLatence
from latence._utils import process_batch_concurrently

NUM_SAMPLES = 50
MAX_CONCURRENCY = 10


def load_samples(n: int = NUM_SAMPLES):
    ds = load_dataset("wandb/RAGTruth-processed", split=f"test[:{n}]")
    samples = []
    for row in ds:
        is_hallucinated = (
            row["hallucination_labels_processed"]["evident_conflict"] > 0
            or row["hallucination_labels_processed"]["baseless_info"] > 0
        )
        samples.append(
            {
                "response_text": row["output"],
                "raw_context": row["context"],
                "query_text": row["query"],
                "label": "hallucinated" if is_hallucinated else "faithful",
            }
        )
    return samples


async def score_one(sample: dict) -> dict:
    label = sample.pop("label", "unknown")
    async with AsyncLatence() as client:
        result = await client.experimental.trace.rag(**sample)
    return {
        "score": result.score,
        "band": result.band,
        "label": label,
        "latency_ms": result.latency_ms,
        "cost_usd": getattr(result, "_cost_usd", None),
        "balance_remaining": getattr(result, "_balance_remaining", None),
    }


async def main():
    api_key = os.environ.get("LATENCE_API_KEY")
    if not api_key:
        print("ERROR: LATENCE_API_KEY not set")
        sys.exit(1)

    print(f"Loading {NUM_SAMPLES} RAGTruth samples...")
    samples = load_samples(NUM_SAMPLES)
    faithful_count = sum(1 for s in samples if s["label"] == "faithful")
    hallucinated_count = sum(1 for s in samples if s["label"] == "hallucinated")
    print(f"  faithful={faithful_count}, hallucinated={hallucinated_count}")

    print(f"\nScoring with max_concurrency={MAX_CONCURRENCY}...")
    t0 = time.time()
    results = await process_batch_concurrently(
        samples, score_one, max_concurrency=MAX_CONCURRENCY
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    successes = [r for r in results if not isinstance(r, Exception)]
    errors = [r for r in results if isinstance(r, Exception)]
    print(f"\nResults: {len(successes)} OK, {len(errors)} errors")

    if errors:
        for e in errors[:5]:
            print(f"  ERROR: {e}")

    if successes:
        faithful_scores = [r["score"] for r in successes if r["label"] == "faithful" and r["score"] is not None]
        hallucinated_scores = [r["score"] for r in successes if r["label"] == "hallucinated" and r["score"] is not None]

        print(f"\nFaithful samples (n={len(faithful_scores)}):")
        if faithful_scores:
            print(f"  mean={sum(faithful_scores)/len(faithful_scores):.3f}, "
                  f"min={min(faithful_scores):.3f}, max={max(faithful_scores):.3f}")

        print(f"Hallucinated samples (n={len(hallucinated_scores)}):")
        if hallucinated_scores:
            print(f"  mean={sum(hallucinated_scores)/len(hallucinated_scores):.3f}, "
                  f"min={min(hallucinated_scores):.3f}, max={max(hallucinated_scores):.3f}")

        bands = Counter(r["band"] for r in successes)
        print(f"\nBand distribution: {dict(bands)}")

        latencies = [r["latency_ms"] for r in successes if r["latency_ms"]]
        if latencies:
            latencies.sort()
            p50 = latencies[len(latencies)//2]
            p95 = latencies[int(len(latencies)*0.95)]
            print(f"Pod latency: p50={p50:.0f}ms, p95={p95:.0f}ms")

    output_path = "/workspace/trace_results.json"
    with open(output_path, "w") as f:
        json.dump({"successes": successes, "error_count": len(errors), "elapsed_s": elapsed}, f, indent=2)
    print(f"\nFull results written to {output_path}")

    return successes


if __name__ == "__main__":
    asyncio.run(main())
