import os
import pickle
import time
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "5"))
INDEX_DIR = "/app/indices"


def load_index(strategy: str) -> tuple:
    index_path = os.path.join(INDEX_DIR, f"{strategy}_index.faiss")
    chunks_path = os.path.join(INDEX_DIR, f"{strategy}_chunks.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"index not found at {index_path} — run embed.py first"
        )

    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"loaded {strategy} index — {index.ntotal} vectors")
    return index, chunks


def embed_query(query: str, client: OpenAI) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    vector = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(vector)
    return vector


def retrieve_chunks(
    query_vector: np.ndarray,
    index: faiss.Index,
    chunks: list,
    top_k: int = TOP_K,
) -> list[dict]:
    D, I = index.search(query_vector, top_k)

    retrieved = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["similarity_score"] = float(score)
        retrieved.append(chunk)

    return retrieved


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        section = chunk["metadata"]["section"]
        filename = chunk["metadata"]["filename"]
        score = chunk["similarity_score"]
        context_blocks.append(
            f"[Context {i+1} | file: {filename} | section: {section} | score: {score:.3f}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a financial analyst assistant. Answer the question below using ONLY the provided context from SEC 10-K filings.

If the context does not contain enough information to answer the question, respond with: "The provided filings do not contain sufficient information to answer this question."

Do not use any knowledge outside of the provided context. Cite which filing and section your answer comes from.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    return prompt


def generate_answer(prompt: str, client: OpenAI) -> str:
    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def run_query(
    query: str,
    index: faiss.Index,
    chunks: list,
    client: OpenAI,
    top_k: int = TOP_K,
) -> dict:
    start_time = time.time()

    query_vector = embed_query(query, client)
    retrieved = retrieve_chunks(query_vector, index, chunks, top_k)
    prompt = build_prompt(query, retrieved)
    answer = generate_answer(prompt, client)

    elapsed = time.time() - start_time

    return {
        "question": query,
        "answer": answer,
        "contexts": [c["text"] for c in retrieved],
        "retrieved_metadata": [c["metadata"] for c in retrieved],
        "similarity_scores": [c["similarity_score"] for c in retrieved],
        "latency_seconds": round(elapsed, 3),
    }


def demo_query(strategy: str = "fixed"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    index, chunks = load_index(strategy)

    test_query = "What are the main risk factors related to cybersecurity mentioned in these filings?"

    print(f"\nstrategy: {strategy}")
    print(f"query: {test_query}\n")
    print("running pipeline...")

    result = run_query(test_query, index, chunks, client)

    print(f"\n{'='*60}")
    print(f"ANSWER:")
    print(result["answer"])
    print(f"\n{'='*60}")
    print(f"RETRIEVED CHUNKS ({len(result['contexts'])}):")
    for i, (ctx, meta, score) in enumerate(zip(
        result["contexts"],
        result["retrieved_metadata"],
        result["similarity_scores"]
    )):
        print(f"\n  chunk {i+1} | {meta['section']} | score: {score:.3f}")
        print(f"  {ctx[:150]}...")

    print(f"\nlatency: {result['latency_seconds']}s")


if __name__ == "__main__":
    print("testing fixed index...")
    demo_query("fixed")
    print("\n\ntesting semantic index...")
    demo_query("semantic")