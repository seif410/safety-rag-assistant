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
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from langchain.chat_models import init_chat_model

from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval")

# retrieve_with_filter already reranks internally (retrieval_k -> top_n), so the
# eval does NOT rerank again. _extract_text flattens block-list message content
# to a plain string (some providers return content blocks, not a bare string).
from app.core.rag_chain import retrieve_with_filter, run_query, _extract_text

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"

# Seconds to pause between questions in the faithfulness loop so the agent +
# judge Groq calls stay under the free-tier per-minute rate limit.
EVAL_SLEEP_SECONDS = 2.0

# Judge is told to reply with only the number; grab the first 1-5 digit anyway
# so stray formatting ("Score: 4", "4/5") still parses.
_SCORE_RE = re.compile(r"[1-5]")

_JUDGE_PROMPT = """Rate how well the generated answer matches the expected answer.
Expected: {expected}
Generated: {generated}
Score 1-5 (1=wrong, 5=perfect). Reply with ONLY the number."""


def load_eval_set(path: Path = DATASET_PATH) -> list[dict]:
    """Load the hand-written Q&A pairs."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def eval_retrieval_precision(eval_set: list[dict]) -> tuple[float, list[dict]]:
    """Fraction of questions whose expected source appears in retrieved results.

    Returns the precision plus per-question detail for the results file.
    """
    logger.info("Retrieval precision: evaluating %d questions", len(eval_set))
    results = []
    for i, item in enumerate(eval_set, 1):
        logger.info("  [%d/%d] retrieving: %.60s", i, len(eval_set), item["question"])
        # No doc_type filter: honest whole-corpus retrieval. retrieve_with_filter
        # returns the already-reranked top_n documents.
        docs = retrieve_with_filter(item["question"])
        retrieved_sources = [doc.metadata.get("filename") for doc in docs]
        hit = item["expected_source"] in retrieved_sources
        logger.info("  [%d/%d] %s", i, len(eval_set), "HIT" if hit else "MISS")
        results.append(
            {
                "question": item["question"],
                "expected_source": item["expected_source"],
                "retrieved_sources": retrieved_sources,
                "hit": hit,
            }
        )
    precision = sum(r["hit"] for r in results) / len(results) if results else 0.0
    logger.info(
        "Retrieval precision: %.1f%% (%d/%d hits)",
        precision * 100,
        sum(r["hit"] for r in results),
        len(results),
    )
    return precision, results


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
    logger.info(
        "Answer faithfulness: evaluating %d questions (agent + judge per question)",
        len(eval_set),
    )
    judge = init_chat_model(
        model=settings.chat_model,
        model_provider=settings.chat_model_provider,
        api_key=settings.google_api_key,
    )
    results = []
    for i, item in enumerate(eval_set):
        logger.info(
            "  [%d/%d] running agent: %.60s", i + 1, len(eval_set), item["question"]
        )
        # Unique session per item so conversational memory from one question does
        # not bleed into the next.
        result = run_query(item["question"], session_id=f"eval-{i}")
        score = _judge_score(judge, item["expected_answer"], result["answer"])
        logger.info("  [%d/%d] judge score: %d/5", i + 1, len(eval_set), score)
        results.append(
            {
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "generated_answer": result["answer"],
                "sources": result["sources"],
                "score": score,
            }
        )
        # Space out Groq calls to stay under the per-minute rate limit.
        time.sleep(EVAL_SLEEP_SECONDS)
    faithfulness = (
        sum(r["score"] for r in results) / (len(results) * 5) if results else 0.0
    )
    logger.info("Answer faithfulness: %.1f%%", faithfulness * 100)
    return faithfulness, results


def save_results(payload: dict) -> Path:
    """Write a timestamped results file to eval/results/ and return its path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> None:
    logger.info("Loading eval dataset from %s", DATASET_PATH)
    eval_set = load_eval_set()
    logger.info("Loaded %d Q&A pairs", len(eval_set))

    precision, retrieval_results = eval_retrieval_precision(eval_set)
    faithfulness, faithfulness_results = eval_answer_faithfulness(eval_set)

    print(f"Retrieval Precision:  {precision:.1%}")
    print(f"Answer Faithfulness:  {faithfulness:.1%}")

    out_path = save_results(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset_size": len(eval_set),
            "metrics": {
                "retrieval_precision": precision,
                "answer_faithfulness": faithfulness,
            },
            "retrieval": retrieval_results,
            "faithfulness": faithfulness_results,
        }
    )
    logger.info("Saved results to %s", out_path)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
