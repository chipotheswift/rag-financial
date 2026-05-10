import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from src.pipeline.rag import load_index, run_query

# pydantic models define the shape of request and response data
# fastapi uses these to automatically validate incoming requests
# and serialize outgoing responses
# if a request doesn't match the model fastapi returns a 422 error
# with a clear explanation of what's wrong

class QueryRequest(BaseModel):
    question: str
    strategy: str = "fixed"  # default to fixed if not specified
    top_k: int = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
    strategy: str
    contexts: list[str]
    similarity_scores: list[float]
    retrieved_sections: list[str]
    latency_seconds: float


# lifespan handles startup and shutdown logic
# we load the indices once at startup rather than on every request
# loading a FAISS index takes ~0.5 seconds — fine once at startup
# but would add 0.5s to every single query if done per-request
# this is called the "warm start" pattern in ML serving

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup — runs once when server starts
    print("loading indices...")
    app.state.fixed_index, app.state.fixed_chunks = load_index("fixed")
    app.state.semantic_index, app.state.semantic_chunks = load_index("semantic")
    app.state.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("indices loaded — server ready")

    yield  # server runs here

    # shutdown — runs when server stops
    print("shutting down")


app = FastAPI(
    title="RAG Financial API",
    description="Baseline RAG pipeline for SEC 10-K filing QA",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    Returns 200 if the server is running.
    Used to verify the server started correctly before running evaluation.
    """
    return {"status": "ok", "indices_loaded": True}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main RAG query endpoint.
    Accepts a question and strategy, returns answer with retrieved context.
    Strategy must be either 'fixed' or 'semantic'.
    """
    if request.strategy not in ("fixed", "semantic"):
        raise HTTPException(
            status_code=400,
            detail="strategy must be 'fixed' or 'semantic'"
        )

    # select the right index based on strategy
    if request.strategy == "fixed":
        index = app.state.fixed_index
        chunks = app.state.fixed_chunks
    else:
        index = app.state.semantic_index
        chunks = app.state.semantic_chunks

    # run the full RAG pipeline
    result = run_query(
        query=request.question,
        index=index,
        chunks=chunks,
        client=app.state.client,
        top_k=request.top_k,
    )

    # extract section names from metadata for the response
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
    )


@app.get("/benchmark/preview")
def benchmark_preview():
    """
    Returns the first question from each tier of the benchmark.
    Quick sanity check that the benchmark loaded correctly.
    """
    from src.evaluation.benchmark import get_tier
    return {
        "tier_1_example": get_tier(1)[0],
        "tier_2_example": get_tier(2)[0],
        "tier_3_example": get_tier(3)[0],
    }