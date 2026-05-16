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
    if questions is None:
        questions = get_benchmark()

    print(f"\n{'='*60}")
    print(f"evaluating strategy: {strategy}")
    print(f"questions: {len(questions)}")
    print(f"{'='*60}\n")

    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        health.raise_for_status()
        print("API health check passed\n")
    except Exception as e:
        print(f"API not reachable: {e}")
        print("make sure the FastAPI server is running")
        return None

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
        time.sleep(0.5)

    print(f"\ncollected {len(results)} results ({len(failed)} failed)")

    if not results:
        print("no results to evaluate")
        return None

    print("\ncomputing RAGAS scores...")
    print("(this makes additional OpenAI API calls — may take 2-3 minutes)\n")

    ragas_data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }

    dataset = Dataset.from_dict(ragas_data)
    ragas_scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    scores_df = ragas_scores.to_pandas()

    for i, result in enumerate(results):
        result["faithfulness"] = float(scores_df.iloc[i]["faithfulness"])
        result["answer_relevancy"] = float(scores_df.iloc[i]["answer_relevancy"])
        result["context_precision"] = float(scores_df.iloc[i]["context_precision"])
        result["context_recall"] = float(scores_df.iloc[i]["context_recall"])

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

    timestamp = int(time.time())
    results_path = os.path.join(RESULTS_DIR, f"{strategy}_results_{timestamp}.json")
    aggregate_path = os.path.join(RESULTS_DIR, f"{strategy}_aggregate_{timestamp}.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\nresults saved → {results_path}")
    print(f"aggregate saved → {aggregate_path}")
    return aggregate


def print_comparison(fixed_agg: dict, semantic_agg: dict):
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


def run_alternative_evaluation(strategy: str = "semantic") -> dict:
    questions = get_benchmark()

    print(f"\n{'='*60}")
    print(f"evaluating ALTERNATIVE pipeline — strategy: {strategy}")
    print(f"questions: {len(questions)}")
    print(f"{'='*60}\n")

    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        health.raise_for_status()
        print("API health check passed\n")
    except Exception as e:
        print(f"API not reachable: {e}")
        return None

    results = []
    failed = []
    low_confidence_count = 0

    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['id']} — {q['question'][:60]}...")
        payload = {
            "question": q["question"],
            "strategy": strategy,
            "use_alternative": True,
        }
        result = None
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  failed: {e}")
                else:
                    time.sleep(2 ** attempt)

        if result is None:
            failed.append(q["id"])
            continue

        if result.get("low_confidence"):
            low_confidence_count += 1

        results.append({
            "question_id": q["id"],
            "tier": q["tier"],
            "question": q["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": q["ground_truth"],
            "similarity_scores": result["similarity_scores"],
            "reranker_scores": result.get("reranker_scores", []),
            "retrieved_sections": result["retrieved_sections"],
            "latency_seconds": result["latency_seconds"],
            "low_confidence": result.get("low_confidence", False),
            "target_section": result.get("target_section"),
            "strategy": f"{strategy}_alternative",
            "filename": q["filename"],
            "expected_section": q["section"],
        })
        time.sleep(0.5)

    print(f"\ncollected {len(results)} results")
    print(f"low confidence triggered: {low_confidence_count} times")

    if not results:
        return None

    print("\ncomputing RAGAS scores...")

    ragas_data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }

    dataset = Dataset.from_dict(ragas_data)
    ragas_scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    scores_df = ragas_scores.to_pandas()

    for i, result in enumerate(results):
        result["faithfulness"] = float(scores_df.iloc[i]["faithfulness"])
        result["answer_relevancy"] = float(scores_df.iloc[i]["answer_relevancy"])
        result["context_precision"] = float(scores_df.iloc[i]["context_precision"])
        result["context_recall"] = float(scores_df.iloc[i]["context_recall"])

    def mean(values):
        valid = [v for v in values if v is not None and v == v]
        return round(sum(valid) / len(valid), 4) if valid else 0.0

    aggregate = {
        "strategy": f"{strategy}_alternative",
        "total_questions": len(results),
        "failed_questions": len(failed),
        "low_confidence_triggered": low_confidence_count,
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

    timestamp = int(time.time())
    results_path = os.path.join(RESULTS_DIR, f"{strategy}_alternative_results_{timestamp}.json")
    aggregate_path = os.path.join(RESULTS_DIR, f"{strategy}_alternative_aggregate_{timestamp}.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\nresults saved → {results_path}")
    return aggregate


def run_ablation_study() -> dict:
    """
    Tests different FAISS retrieval pool sizes.
    Measures Context Precision and latency at each point.
    Runs Tier 1 + Tier 2 only — Tier 3 doesn't benefit
    from pool size changes since answers don't exist anyway.
    """
    from src.evaluation.benchmark import get_tier

    questions = get_tier(1) + get_tier(2)
    pool_sizes = [5, 10, 15, 20, 30, 50]
    results = {}

    print("\n" + "="*60)
    print("FAISS RETRIEVAL POOL SIZE ABLATION")
    print(f"questions: {len(questions)} (Tier 1 + Tier 2)")
    print(f"pool sizes: {pool_sizes}")
    print("="*60)

    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        health.raise_for_status()
        print("API health check passed\n")
    except Exception as e:
        print(f"API not reachable: {e}")
        return None

    for pool_size in pool_sizes:
        print(f"\ntesting pool_size={pool_size}...")
        latencies = []
        answers = []
        contexts_list = []
        ground_truths = []
        failed = 0

        for q in questions:
            payload = {
                "question": q["question"],
                "strategy": "semantic",
                "use_alternative": True,
                "top_k_retrieval": pool_size,
            }
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
                latencies.append(result["latency_seconds"])
                answers.append(result["answer"])
                contexts_list.append(result["contexts"])
                ground_truths.append(q["ground_truth"])
            except Exception as e:
                print(f"  failed: {e}")
                failed += 1
            time.sleep(0.3)

        if not answers:
            continue

        print(f"  computing RAGAS for pool_size={pool_size}...")
        ragas_data = {
            "question": [q["question"] for q in questions[:len(answers)]],
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(ragas_data)
        ragas_scores = evaluate(
            dataset,
            metrics=[context_precision, context_recall],
        )
        scores_df = ragas_scores.to_pandas()

        avg_precision = round(float(scores_df["context_precision"].mean()), 4)
        avg_recall = round(float(scores_df["context_recall"].mean()), 4)
        avg_latency = round(sum(latencies) / len(latencies), 3)

        results[pool_size] = {
            "pool_size": pool_size,
            "avg_context_precision": avg_precision,
            "avg_context_recall": avg_recall,
            "avg_latency_seconds": avg_latency,
            "failed": failed,
        }

        print(f"  pool={pool_size} | precision={avg_precision} | recall={avg_recall} | latency={avg_latency}s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "ablation_pool_size.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nablation results saved → {path}")
    print(f"\n{'Pool Size':<12} {'Precision':<12} {'Recall':<12} {'Latency':<12}")
    print("-" * 48)
    for size, r in sorted(results.items()):
        print(f"{size:<12} {r['avg_context_precision']:<12} {r['avg_context_recall']:<12} {r['avg_latency_seconds']:<12}")

    return results


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"

    if mode == "baseline":
        fixed_results = run_evaluation("fixed")
        semantic_results = run_evaluation("semantic")
        if fixed_results and semantic_results:
            print_comparison(fixed_results, semantic_results)

    elif mode == "alternative":
        alt_results = run_alternative_evaluation("semantic")
        if alt_results:
            print("\nAlternative pipeline aggregate:")
            for k, v in alt_results.items():
                if k != "by_tier":
                    print(f"  {k}: {v}")
            print("\nBy tier:")
            for tier, metrics in alt_results["by_tier"].items():
                print(f"  {tier}: {metrics}")

    elif mode == "ablation":
        run_ablation_study()