import os
import time
import wandb
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = "rag-financial"


def init_run(run_name: str, config: dict) -> wandb.run:
    """
    Initializes a W&B run with a given name and config dict.
    Config stores all the hyperparameters and settings for this run
    so you can reproduce it exactly later.
    Every pipeline execution gets its own run — fixed and semantic
    are separate runs so you can compare them side by side in W&B.
    """
    run = wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        config=config,
        reinit=True,
    )
    return run


def log_chunking_stats(strategy: str, chunks: list):
    """
    Logs chunk-level statistics for one strategy to W&B.
    Called after chunking, before embedding.
    """
    from collections import Counter

    section_counts = Counter(
        c["metadata"]["section"] for c in chunks
    )

    chunk_sizes = [c["metadata"]["chunk_size_chars"] for c in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0

    wandb.log({
        f"{strategy}/total_chunks": len(chunks),
        f"{strategy}/avg_chunk_size_chars": round(avg_size, 1),
        f"{strategy}/chunks_section_1A": section_counts.get("section_1A", 0),
        f"{strategy}/chunks_section_1": section_counts.get("section_1", 0),
        f"{strategy}/chunks_section_7": section_counts.get("section_7", 0),
        f"{strategy}/chunks_section_8": section_counts.get("section_8", 0),
    })

    print(f"logged chunking stats for {strategy}")


def log_embedding_stats(
    strategy: str,
    num_chunks: int,
    elapsed_seconds: float,
    estimated_cost: float,
):
    """
    Logs embedding pipeline stats — cost, time, throughput.
    Called after embed_chunks() completes.
    """
    wandb.log({
        f"{strategy}/embedding_time_seconds": round(elapsed_seconds, 2),
        f"{strategy}/embedding_cost_usd": round(estimated_cost, 4),
        f"{strategy}/chunks_per_second": round(num_chunks / elapsed_seconds, 1),
    })

    print(f"logged embedding stats for {strategy}")


def log_index_stats(strategy: str, index):
    """
    Logs FAISS index metadata.
    index.ntotal = number of vectors stored.
    index.d = dimensionality of each vector.
    """
    wandb.log({
        f"{strategy}/index_total_vectors": index.ntotal,
        f"{strategy}/index_dimensions": index.d,
    })

    print(f"logged index stats for {strategy}")


def log_query_result(result: dict, strategy: str, question_id: int):
    """
    Logs one query result to W&B.
    Called after run_query() for each benchmark question.
    Logs the answer, retrieved contexts, similarity scores, and latency.
    """
    wandb.log({
        f"{strategy}/q{question_id}_latency": result["latency_seconds"],
        f"{strategy}/q{question_id}_top1_score": result["similarity_scores"][0],
        f"{strategy}/q{question_id}_top5_score": result["similarity_scores"][-1],
        f"{strategy}/q{question_id}_score_range": (
            result["similarity_scores"][0] - result["similarity_scores"][-1]
        ),
    })


def log_ragas_scores(scores: dict, strategy: str, question_id: int):
    """
    Logs RAGAS evaluation scores for one question.
    Called after RAGAS evaluates the answer.
    scores dict has keys: faithfulness, answer_relevancy,
    context_precision, context_recall.
    """
    wandb.log({
        f"{strategy}/q{question_id}_faithfulness": scores.get("faithfulness", 0),
        f"{strategy}/q{question_id}_answer_relevancy": scores.get("answer_relevancy", 0),
        f"{strategy}/q{question_id}_context_precision": scores.get("context_precision", 0),
        f"{strategy}/q{question_id}_context_recall": scores.get("context_recall", 0),
    })


def finish_run():
    """
    Closes the W&B run cleanly.
    Always call this at the end of a pipeline run.
    Without it W&B may not sync all data before the process exits.
    """
    wandb.finish()
    print("W&B run finished and synced")


def test_logging():
    """
    Runs a quick test to verify W&B connection works.
    Logs dummy data for both strategies to confirm the dashboard
    is receiving data correctly.
    """
    print("testing W&B connection...")

    run = init_run(
        run_name="connection-test",
        config={
            "embedding_model": os.getenv("EMBEDDING_MODEL"),
            "generator_model": os.getenv("GENERATOR_MODEL"),
            "chunk_size": 512,
            "chunk_overlap": 50,
            "similarity_threshold": 0.5,
            "top_k": int(os.getenv("TOP_K", 5)),
            "dev_filings": 10,
            "test": True,
        }
    )

    # log dummy metrics to verify connection
    wandb.log({
        "fixed/total_chunks": 4877,
        "semantic/total_chunks": 3700,
        "fixed/embedding_cost_usd": 0.0195,
        "semantic/embedding_cost_usd": 0.0148,
        "fixed/embedding_time_seconds": 53.0,
        "semantic/embedding_time_seconds": 27.0,
    })

    print(f"run url: {run.url}")
    finish_run()
    print("W&B test passed — check your dashboard at wandb.ai")


if __name__ == "__main__":
    test_logging()