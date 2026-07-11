"""Offline evaluation for the Safety RAG Assistant.

Two metrics over a hand-written Q&A set (``eval_dataset.json``):

- **Retrieval precision** — for each question, does the expected source document
  appear among the retrieved+reranked chunks? Uses the general (no doc_type
  filter) retrieval path, so it measures end-to-end retrieval quality.
- **Answer faithfulness** — LLM-as-judge scores how well the generated answer
  matches the expected answer (1-5), normalized to 0-1.

Requires the full stack to be reachable: Qdrant running with the corpus already
ingested, plus NVIDIA / Cohere / Google API keys in ``.env``. Answer faithfulness
runs the full agent once per question, so it costs Gemini + Cohere calls.

Run from the project root:

    python -m eval.run_eval
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

from langchain.chat_models import init_chat_model

from app.config import settings
from logger import log_header, log_info, log_success, log_warning

# retrieve_with_filter reranks internally (no second rerank here). _extract_text
# flattens block-list message content some providers return into a plain string.
from app.core.rag_chain import retrieve_with_filter, run_query, _extract_text

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"

# Pause between questions so Gemini calls stay under the per-minute rate limit.
EVAL_SLEEP_SECONDS = 2.0

# Grab first 1-5 digit so stray formatting ("Score: 4", "4/5") still parses.
_SCORE_RE = re.compile(r"[1-5]")

_JUDGE_PROMPT = """Rate how well the generated answer matches the expected answer.
Expected: {expected}
Generated: {generated}
Score 1-5 (1=wrong, 5=perfect). Reply with ONLY the number."""


def load_eval_set(path: Path = DATASET_PATH) -> list[dict]:
    """Load the hand-written Q&A pairs."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def eval_retrieval(eval_set: list[dict], ks: tuple[int, ...] = (1, 3, 5)):
    """Retrieval quality via hit-rate@k and MRR — no LLM/judge calls.

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


def _judge_score(judge, expected: str, generated: str) -> int:
    """LLM-as-judge: score generated vs expected answer on 1-5 (0 if unparseable)."""
    judgment = judge.invoke(
        [
            {
                "role": "user",
                "content": _JUDGE_PROMPT.format(expected=expected, generated=generated),
            }
        ]
    )
    text = _extract_text(judgment.content).strip()
    match = _SCORE_RE.search(text)
    return int(match.group()) if match else 0


def eval_answer_faithfulness(eval_set: list[dict]) -> tuple[float, list[dict]]:
    """LLM-as-judge answer quality, normalized to 0-1, plus per-question detail."""
    log_header(
        f"Answer faithfulness: evaluating {len(eval_set)} questions "
        "(agent + judge per question)"
    )
    judge = init_chat_model(
        model=settings.chat_model,
        model_provider=settings.chat_model_provider,
        api_key=settings.google_api_key,
    )
    results = []
    for i, item in enumerate(eval_set):
        log_info(f"[{i + 1}/{len(eval_set)}] running agent: {item['question'][:60]}")
        # Unique session per item so memory doesn't bleed between questions.
        result = run_query(item["question"], session_id=f"eval-{i}")
        score = _judge_score(judge, item["expected_answer"], result["answer"])
        log_info(f"[{i + 1}/{len(eval_set)}] judge score: {score}/5")
        results.append(
            {
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "generated_answer": result["answer"],
                "sources": result["sources"],
                "score": score,
            }
        )
        # Space out Gemini calls to stay under the per-minute rate limit.
        time.sleep(EVAL_SLEEP_SECONDS)
    faithfulness = (
        sum(r["score"] for r in results) / (len(results) * 5) if results else 0.0
    )
    log_success(f"Answer faithfulness: {faithfulness * 100:.1f}%")
    return faithfulness, results


def save_results(payload: dict) -> Path:
    """Write a timestamped results file to eval/results/ and return its path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def main(with_faithfulness: bool = False) -> None:
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

    # Faithfulness hits the Gemini quota — only when explicitly requested.
    if with_faithfulness:
        faithfulness, faithfulness_results = eval_answer_faithfulness(eval_set)
        payload["metrics"]["answer_faithfulness"] = faithfulness
        payload["faithfulness"] = faithfulness_results
        print(f"  {'faithfulness':12s} {faithfulness:.1%}")

    out_path = save_results(payload)
    log_success(f"Saved results to {out_path}")


if __name__ == "__main__":
    import sys

    main(with_faithfulness="--faithfulness" in sys.argv)
