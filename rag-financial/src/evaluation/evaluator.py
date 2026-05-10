import json
import os
import time
import requests
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from src.evaluation.benchmark import get_benchmark

load_dotenv()

API_URL = "http://localhost:8000"
RESULTS_DIR = "/app/data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def query_api(question: str, strategy: str, top_k: int = 5) -> dict:
    """
    Sends one question to the FastAPI endpoint and returns the result.
    Retries up to 3 times on failure with exponential backoff.
    """
    payload = {
        "question": question,
        "strategy": strategy,
        "top_k": top_k,
    }

    for attempt in range(3):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == 2:
                print(f"  failed after 3 attempts: {e}")
                return None
            time.sleep(2 ** attempt)


def run_evaluation(strategy: str, questions: list = None) -> dict:
    """
    Runs the full evaluation loop for one strategy.
    Sends every benchmark question through the API,
    collects answers and contexts, then computes RAGAS scores.
    Returns a dict with all results and aggregate metrics.
    """
    if questions is None:
        questions = get_benchmark()

    print(f"\n{'='*60}")
    print(f"evaluating strategy: {strategy}")
    print(f"questions: {len(questions)}")
    print(f"{'='*60}\n")

    # check API is reachable before starting
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        health.raise_for_status()
        print("API health check passed\n")
    except Exception as e:
        print(f"API not reachable: {e}")
        print("make sure the FastAPI server is running")
        return None

    # collect results for all questions
    results = []
    failed = []

    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['id']} — {q['question'][:60]}...")

        result = query_api(q["question"], strategy)

        if result is None:
            failed.append(q["id"])
            continue

        results.append({
            "question_id": q["id"],
            "tier": q["tier"],
            "question": q["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": q["ground_truth"],
            "similarity_scores": result["similarity_scores"],
            "retrieved_sections": result["retrieved_sections"],
            "latency_seconds": result["latency_seconds"],
            "strategy": strategy,
            "filename": q["filename"],
            "expected_section": q["section"],
        })

        # small delay to avoid hammering the API
        time.sleep(0.5)

    print(f"\ncollected {len(results)} results ({len(failed)} failed)")

    if not results:
        print("no results to evaluate")
        return None

    # compute RAGAS scores
    print("\ncomputing RAGAS scores...")
    print("(this makes additional OpenAI API calls — may take 2-3 minutes)\n")

    # RAGAS expects a HuggingFace Dataset with specific column names
    ragas_data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }

    dataset = Dataset.from_dict(ragas_data)

    # run RAGAS evaluation
    ragas_scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # convert to dict for easy access
    scores_df = ragas_scores.to_pandas()

    # attach RAGAS scores back to individual results
    for i, result in enumerate(results):
        result["faithfulness"] = float(scores_df.iloc[i]["faithfulness"])
        result["answer_relevancy"] = float(scores_df.iloc[i]["answer_relevancy"])
        result["context_precision"] = float(scores_df.iloc[i]["context_precision"])
        result["context_recall"] = float(scores_df.iloc[i]["context_recall"])

    # compute aggregate metrics
    def mean(values):
        valid = [v for v in values if v is not None and v == v]
        return round(sum(valid) / len(valid), 4) if valid else 0.0

    aggregate = {
        "strategy": strategy,
        "total_questions": len(results),
        "failed_questions": len(failed),
        "avg_faithfulness": mean([r["faithfulness"] for r in results]),
        "avg_answer_relevancy": mean([r["answer_relevancy"] for r in results]),
        "avg_context_precision": mean([r["context_precision"] for r in results]),
        "avg_context_recall": mean([r["context_recall"] for r in results]),
        "avg_latency_seconds": mean([r["latency_seconds"] for r in results]),
        "by_tier": {
            "tier_1": {
                "avg_faithfulness": mean([r["faithfulness"] for r in results if r["tier"] == 1]),
                "avg_context_precision": mean([r["context_precision"] for r in results if r["tier"] == 1]),
            },
            "tier_2": {
                "avg_faithfulness": mean([r["faithfulness"] for r in results if r["tier"] == 2]),
                "avg_context_precision": mean([r["context_precision"] for r in results if r["tier"] == 2]),
            },
            "tier_3": {
                "avg_faithfulness": mean([r["faithfulness"] for r in results if r["tier"] == 3]),
                "avg_context_precision": mean([r["context_precision"] for r in results if r["tier"] == 3]),
            },
        }
    }

    # save full results to disk
    timestamp = int(time.time())
    results_path = os.path.join(
        RESULTS_DIR, f"{strategy}_results_{timestamp}.json"
    )
    aggregate_path = os.path.join(
        RESULTS_DIR, f"{strategy}_aggregate_{timestamp}.json"
    )

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\nresults saved → {results_path}")
    print(f"aggregate saved → {aggregate_path}")

    return aggregate


def print_comparison(fixed_agg: dict, semantic_agg: dict):
    """
    Prints a side by side comparison of fixed vs semantic metrics.
    This is your core research finding in table form.
    """
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON — FIXED vs SEMANTIC")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Fixed':>10} {'Semantic':>10} {'Delta':>10}")
    print(f"{'-'*60}")

    metrics = [
        ("Faithfulness", "avg_faithfulness"),
        ("Answer Relevancy", "avg_answer_relevancy"),
        ("Context Precision", "avg_context_precision"),
        ("Context Recall", "avg_context_recall"),
        ("Avg Latency (s)", "avg_latency_seconds"),
    ]

    for label, key in metrics:
        fixed_val = fixed_agg.get(key, 0)
        sem_val = semantic_agg.get(key, 0)
        delta = sem_val - fixed_val
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        print(f"{label:<30} {fixed_val:>10.4f} {sem_val:>10.4f} {delta_str:>10}")

    print(f"\nBy tier — Faithfulness:")
    for tier in ["tier_1", "tier_2", "tier_3"]:
        f_val = fixed_agg["by_tier"][tier]["avg_faithfulness"]
        s_val = semantic_agg["by_tier"][tier]["avg_faithfulness"]
        delta = s_val - f_val
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        print(f"  {tier:<28} {f_val:>10.4f} {s_val:>10.4f} {delta_str:>10}")


if __name__ == "__main__":
    # run fixed evaluation first
    fixed_results = run_evaluation("fixed")

    # then semantic
    semantic_results = run_evaluation("semantic")

    # print comparison
    if fixed_results and semantic_results:
        print_comparison(fixed_results, semantic_results)