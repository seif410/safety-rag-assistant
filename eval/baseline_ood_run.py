"""Baseline capture of the prebuilt-agent behaviour on out-of-scope questions.

Throwaway instrumentation, not a grader. Runs the current agent over the OOD and
scope datasets and dumps raw answers for manual labelling, so the same questions
can be re-run after the LangGraph migration and compared side by side.

Each question gets its own session id: shared memory would let an earlier refusal
condition the next answer and invalidate the comparison. Results are rewritten
after every item so a crash mid-run keeps the calls already paid for.

Requires Qdrant running with the corpus ingested, plus the API keys in ``.env``.
Spends real quota (18 LLM + rerank calls, ~3 minutes with the rate-limit sleep).

Run from the project root:

    python -m eval.baseline_ood_run
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.core.rag_chain import run_query
from logger import log_error, log_header, log_info, log_success

EVAL_DIR = Path(__file__).resolve().parent
OOD_DATASET_PATH = EVAL_DIR / "eval_dataset_ood.json"
SCOPE_DATASET_PATH = EVAL_DIR / "eval_dataset_scope.json"
RESULTS_DIR = EVAL_DIR / "results"
BASELINE_TAG = "v0.1-prebuilt-agent"
SLEEP_SECONDS = 8  # Cohere trial keys allow 10 rerank calls/min


def load_dataset(path: Path) -> list[dict]:
    """Load one hand-written out-of-scope question set."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def baseline_sha() -> str:
    """Short SHA of the tag marking the pre-migration architecture."""
    completed = subprocess.run(
        ["git", "rev-parse", "--short", BASELINE_TAG],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def capture_answer(item: dict) -> dict:
    """Ask the agent one question and build its (unlabelled) result record.

    A failed question is recorded as its error string rather than raised: losing
    the remaining 17 paid-for calls to one transient API error is worse than a
    hole in the run. ``outcome`` and ``citation_valid`` are labelled by hand.
    """
    start = time.perf_counter()
    try:
        response = run_query(item["question"], session_id=f"baseline-{item['id']}")
        answer = response["answer"]
        sources = response["sources"]
    except Exception as exc:
        answer = f"ERROR: {exc}"
        sources = []
        log_error(f"  failed: {exc}")

    return {
        "id": item["id"],
        "question": item["question"],
        "expected": item["expected_behavior"],
        "answer": answer,
        "sources": sources,
        "latency_s": round(time.perf_counter() - start, 2),
        "outcome": None,
        "citation_valid": None,
    }


def save_report(report: dict, path: Path) -> None:
    """Overwrite the results file with everything captured so far."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main() -> None:
    eval_set = load_dataset(OOD_DATASET_PATH) + load_dataset(SCOPE_DATASET_PATH)
    log_header(f"Baseline OOD run: {len(eval_set)} questions")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()
    out_path = RESULTS_DIR / f"baseline_ood_{started_at:%Y%m%d_%H%M%S}.json"

    report = {
        "run_metadata": {
            "timestamp": started_at.isoformat(timespec="seconds"),
            "chat_model": settings.chat_model,
            "chat_temperature": settings.chat_temperature,
            "baseline_tag": BASELINE_TAG,
            "baseline_sha": baseline_sha(),
        },
        "results": [],
    }

    for i, item in enumerate(eval_set, 1):
        log_info(f"[{i}/{len(eval_set)}] {item['id']}: {item['question'][:70]}")
        report["results"].append(capture_answer(item))
        save_report(report, out_path)
        if i < len(eval_set):
            time.sleep(SLEEP_SECONDS)

    log_success(f"Wrote {len(report['results'])} results to {out_path}")


if __name__ == "__main__":
    main()
