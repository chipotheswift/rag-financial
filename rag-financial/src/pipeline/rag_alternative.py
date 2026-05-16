import os
import time
import pickle
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")
TOP_K_FINAL = 5
INDEX_DIR = "/app/indices"

CONFIDENCE_THRESHOLD = 0.30
GAP_THRESHOLD = 0.05

SECTION_ROUTING = {
    "risk": "section_1A",
    "cybersecurity": "section_1A",
    "investment": "section_1A",
    "danger": "section_1A",
    "threat": "section_1A",
    "business": "section_1",
    "overview": "section_1",
    "company": "section_1",
    "mission": "section_1",
    "product": "section_1",
    "revenue": "section_7",
    "financial": "section_7",
    "earnings": "section_7",
    "profit": "section_7",
    "performance": "section_7",
    "discussion": "section_7",
    "management": "section_7",
}

CROSS_SECTION_KEYWORDS = [
    "and what", "relate to", "connect", "both", "also",
    "how does", "what do", "which companies", "compare"
]


def load_cross_encoder() -> CrossEncoder:
    print("loading cross-encoder...")
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("cross-encoder loaded")
    return model


def detect_target_section(question: str) -> str | None:
    question_lower = question.lower()
    for keyword in CROSS_SECTION_KEYWORDS:
        if keyword in question_lower:
            return None
    for keyword, section in SECTION_ROUTING.items():
        if keyword in question_lower:
            return section
    return None


def filter_chunks_by_section(
    chunks: list[dict],
    section: str | None,
) -> tuple[list[dict], list[int]]:
    if section is None:
        return chunks, list(range(len(chunks)))

    filtered = []
    indices = []
    for i, chunk in enumerate(chunks):
        if chunk["metadata"]["section"] == section:
            filtered.append(chunk)
            indices.append(i)

    if len(filtered) < 10:
        return chunks, list(range(len(chunks)))

    return filtered, indices


def detect_low_confidence(scores: list[float]) -> bool:
    if len(scores) < 2:
        return False
    top_score = scores[0]
    gap = scores[0] - scores[1]
    return top_score < CONFIDENCE_THRESHOLD and gap < GAP_THRESHOLD


def rerank_chunks(
    query: str,
    chunks: list[dict],
    cross_encoder: CrossEncoder,
    top_k: int = TOP_K_FINAL,
) -> list[dict]:
    if not chunks:
        return []
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["reranker_score"] = float(score)
    reranked = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)
    return reranked[:top_k]


def embed_query(query: str, client: OpenAI) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    vector = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(vector)
    return vector


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        section = chunk["metadata"]["section"]
        filename = chunk["metadata"]["filename"]
        score = chunk.get("reranker_score", chunk.get("similarity_score", 0))
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
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def run_query_alternative(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    client: OpenAI,
    cross_encoder: CrossEncoder,
    top_k_retrieval: int = 15, 
) -> dict:
    start_time = time.time()

    target_section = detect_target_section(query)
    filtered_chunks, filtered_indices = filter_chunks_by_section(
        chunks, target_section
    )
    query_vector = embed_query(query, client)

    if target_section and len(filtered_chunks) < len(chunks):
        filtered_embeddings = np.zeros(
            (len(filtered_chunks), index.d), dtype=np.float32
        )
        for new_idx, orig_idx in enumerate(filtered_indices):
            index.reconstruct(orig_idx, filtered_embeddings[new_idx])
        sub_index = faiss.IndexFlatIP(index.d)
        sub_index.add(filtered_embeddings)
        search_index = sub_index
        search_chunks = filtered_chunks
    else:
        search_index = index
        search_chunks = chunks

    # use the passed-in top_k_retrieval instead of hardcoded constant
    k = min(top_k_retrieval, search_index.ntotal)
    D, I = search_index.search(query_vector, k)

    candidates = []
    similarity_scores = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        chunk = search_chunks[idx].copy()
        chunk["similarity_score"] = float(score)
        candidates.append(chunk)
        similarity_scores.append(float(score))

    low_confidence = detect_low_confidence(similarity_scores)

    if low_confidence:
        elapsed = time.time() - start_time
        return {
            "question": query,
            "answer": "The provided filings do not contain sufficient information to answer this question.",
            "contexts": [c["text"] for c in candidates[:TOP_K_FINAL]],
            "retrieved_metadata": [c["metadata"] for c in candidates[:TOP_K_FINAL]],
            "similarity_scores": similarity_scores[:TOP_K_FINAL],
            "reranker_scores": [],
            "latency_seconds": round(elapsed, 3),
            "low_confidence": True,
            "target_section": target_section,
        }

    reranked = rerank_chunks(query, candidates, cross_encoder, TOP_K_FINAL)
    prompt = build_prompt(query, reranked)
    answer = generate_answer(prompt, client)
    elapsed = time.time() - start_time

    return {
        "question": query,
        "answer": answer,
        "contexts": [c["text"] for c in reranked],
        "retrieved_metadata": [c["metadata"] for c in reranked],
        "similarity_scores": similarity_scores[:TOP_K_FINAL],
        "reranker_scores": [c["reranker_score"] for c in reranked],
        "latency_seconds": round(elapsed, 3),
        "low_confidence": False,
        "target_section": target_section,
    }


def demo_alternative(strategy: str = "semantic"):
    from src.pipeline.rag import load_index
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    cross_encoder = load_cross_encoder()
    index, chunks = load_index(strategy)
    test_cases = [
        "What cybersecurity risks does Heritage Financial face?",
        "What was the exact revenue figure for Arcimoto in fiscal year 2018?",
    ]
    for query in test_cases:
        print(f"\n{'='*60}")
        print(f"query: {query}")
        section = detect_target_section(query)
        print(f"target section: {section}")
        result = run_query_alternative(query, index, chunks, client, cross_encoder)
        print(f"low confidence: {result['low_confidence']}")
        print(f"latency: {result['latency_seconds']}s")
        print(f"\nanswer:\n{result['answer']}")
        if result["reranker_scores"]:
            print(f"\nreranker scores: {[round(s,3) for s in result['reranker_scores']]}")
            print(f"similarity scores: {[round(s,3) for s in result['similarity_scores']]}")


if __name__ == "__main__":
    demo_alternative("semantic")