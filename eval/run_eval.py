"""Offline evaluation for the Safety RAG Assistant.

Retrieval metrics over a hand-written Q&A set (``eval_dataset.json``), no LLM
calls: for each question, where does the expected source document rank among the
retrieved+reranked chunks? Reported as hit-rate@k (k=1,3,5) and MRR. Uses the
general (no doc_type filter) retrieval path, so it measures end-to-end retrieval
quality.

Requires Qdrant running with the corpus already ingested, plus the NVIDIA /
Cohere API keys in ``.env``.

Run from the project root:

    python -m eval.run_eval
"""

import json
from datetime import datetime
from pathlib import Path

from logger import log_header, log_info, log_success, log_warning

# retrieve_with_filter reranks internally (no second rerank here).
from app.core.rag_chain import retrieve_with_filter

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"


def load_eval_set(path: Path = DATASET_PATH) -> list[dict]:
    """Load the hand-written Q&A pairs."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def eval_retrieval(eval_set: list[dict], ks: tuple[int, ...] = (1, 3, 5)):
    """Retrieval quality via hit-rate@k and MRR — no LLM calls.

    hit-rate@k: fraction of questions whose expected source appears in top-k.
    MRR: mean reciprocal rank of the first chunk from the expected source
    (0 if absent). Whole-corpus path (no doc_type filter) for honest end-to-end.
    """
    log_header(f"Retrieval eval: {len(eval_set)} questions")
    results = []
    for i, item in enumerate(eval_set, 1):
        docs = retrieve_with_filter(item["question"])
        ranked_sources = [doc.metadata.get("filename") for doc in docs]
        expected = item["expected_source"]
        # 1-indexed rank of the first retrieved chunk from the expected source.
        rank = next(
            (r for r, src in enumerate(ranked_sources, 1) if src == expected),
            None,
        )
        hits = {f"hit@{k}": (rank is not None and rank <= k) for k in ks}
        log = log_success if rank else log_warning
        log(
            f"[{i}/{len(eval_set)}] {'HIT' if rank else 'MISS'}  "
            f"rank={rank}  {item['question'][:55]}"
        )
        results.append(
            {
                "question": item["question"],
                "expected_source": expected,
                "retrieved_sources": ranked_sources,
                "rank": rank,
                "reciprocal_rank": (1.0 / rank) if rank else 0.0,
                **hits,
            }
        )
    n = len(results)
    metrics = (
        {f"hit_rate@{k}": sum(r[f"hit@{k}"] for r in results) / n for k in ks}
        if n
        else {}
    )
    metrics["mrr"] = sum(r["reciprocal_rank"] for r in results) / n if n else 0.0
    for name, val in metrics.items():
        log_info(f"Retrieval {name}: {val:.3f}")
    return metrics, results


def save_results(payload: dict) -> Path:
    """Write a timestamped results file to eval/results/ and return its path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> None:
    log_info(f"Loading eval dataset from {DATASET_PATH}")
    eval_set = load_eval_set()
    log_success(f"Loaded {len(eval_set)} Q&A pairs")

    metrics, retrieval_results = eval_retrieval(eval_set)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_size": len(eval_set),
        "metrics": {"retrieval": metrics},
        "retrieval": retrieval_results,
    }

    print("Retrieval metrics:")
    for k, v in metrics.items():
        print(
            f"  {k:12s} {v:.1%}" if k.startswith("hit_rate") else f"  {k:12s} {v:.3f}"
        )

    out_path = save_results(payload)
    log_success(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
