import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from src.pipeline.rag import load_index, run_query
from src.pipeline.rag_alternative import (
    load_cross_encoder,
    run_query_alternative,
)


class QueryRequest(BaseModel):
    question: str
    strategy: str = "fixed"
    top_k: int = 5
    use_alternative: bool = False
    top_k_retrieval: int = 20  # ← new field for ablation


class QueryResponse(BaseModel):
    question: str
    answer: str
    strategy: str
    contexts: list[str]
    similarity_scores: list[float]
    retrieved_sections: list[str]
    latency_seconds: float
    low_confidence: bool = False
    target_section: str | None = None
    reranker_scores: list[float] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("loading indices...")
    app.state.fixed_index, app.state.fixed_chunks = load_index("fixed")
    app.state.semantic_index, app.state.semantic_chunks = load_index("semantic")
    app.state.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("loading cross-encoder...")
    app.state.cross_encoder = load_cross_encoder()
    print("all models loaded — server ready")
    yield
    print("shutting down")


app = FastAPI(
    title="RAG Financial API",
    description="RAG pipeline for SEC 10-K filing QA — baseline and alternative",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "ok", "indices_loaded": True}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if request.strategy not in ("fixed", "semantic"):
        raise HTTPException(
            status_code=400,
            detail="strategy must be 'fixed' or 'semantic'"
        )

    if request.strategy == "fixed":
        index = app.state.fixed_index
        chunks = app.state.fixed_chunks
    else:
        index = app.state.semantic_index
        chunks = app.state.semantic_chunks

    if request.use_alternative:
        result = run_query_alternative(
            query=request.question,
            index=index,
            chunks=chunks,
            client=app.state.client,
            cross_encoder=app.state.cross_encoder,
            top_k_retrieval=request.top_k_retrieval,  # ← passed through
        )
    else:
        result = run_query(
            query=request.question,
            index=index,
            chunks=chunks,
            client=app.state.client,
            top_k=request.top_k,
        )

    sections = [
        m.get("section", "unknown")
        for m in result["retrieved_metadata"]
    ]

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        strategy=request.strategy,
        contexts=result["contexts"],
        similarity_scores=result["similarity_scores"],
        retrieved_sections=sections,
        latency_seconds=result["latency_seconds"],
        low_confidence=result.get("low_confidence", False),
        target_section=result.get("target_section"),
        reranker_scores=result.get("reranker_scores", []),
    )


@app.get("/benchmark/preview")
def benchmark_preview():
    from src.evaluation.benchmark import get_tier
    return {
        "tier_1_example": get_tier(1)[0],
        "tier_2_example": get_tier(2)[0],
        "tier_3_example": get_tier(3)[0],
    }