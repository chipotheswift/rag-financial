import json
import os
import time
import pickle
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
from tqdm import tqdm

load_dotenv()

from src.chunking.fixed import chunk_filing_fixed
from src.chunking.semantic import chunk_filing_semantic, SentenceTransformer

# OpenAI embedding model — text-embedding-3-small outputs 1536-dim vectors
# cheapest OpenAI embedding model, good quality for retrieval tasks
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536

# how many chunks to send per API call
# OpenAI allows up to 2048 inputs per request
# 100 is conservative — avoids hitting token limits on long chunks
BATCH_SIZE = 100

# how many filings to process in this dev run
# keep small until we verify everything works
DEV_FILINGS = 10


def embed_chunks(chunks: list[dict], client: OpenAI) -> np.ndarray:
    """
    Takes a list of chunk dicts and returns a numpy array of embeddings.
    Shape: (num_chunks, EMBEDDING_DIM) — one row per chunk.
    Batches requests to avoid hitting API rate limits.
    """
    all_embeddings = []
    texts = [c["text"] for c in chunks]

    # process in batches
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="embedding batches"):
        batch = texts[i: i + BATCH_SIZE]

        # retry logic — embedding API occasionally returns errors
        # under load, simple retry handles transient failures
        for attempt in range(3):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                # response.data is a list of embedding objects
                # each has a .embedding attribute — the actual vector
                batch_embeddings = [e.embedding for e in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                print(f"  retry {attempt + 1} after error: {e}")
                time.sleep(2 ** attempt)  # exponential backoff

    # convert to numpy array — shape (num_chunks, 1536)
    return np.array(all_embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS flat index from a numpy array of embeddings.
    FlatIP = flat index using inner product (equivalent to cosine
    similarity when vectors are normalized, which we do below).
    Flat means exact search — no approximation.
    For your scale (~10k chunks) exact search is fast enough.
    At million-scale you'd switch to an approximate index (IVF).
    """
    # normalize vectors so inner product = cosine similarity
    faiss.normalize_L2(embeddings)

    # build the index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    # add all vectors to the index
    index.add(embeddings)

    return index


def save_index(
    index: faiss.Index,
    chunks: list[dict],
    strategy: str,
    output_dir: str = "/app/indices"
):
    """
    Saves the FAISS index and chunk metadata to disk.
    We save both because FAISS only stores vectors — not the original
    text. When you retrieve chunk #42 from FAISS you need the metadata
    to know what text and section that chunk came from.
    So we save:
    - index.faiss  → the vectors, searchable by FAISS
    - chunks.pkl   → the original text and metadata, indexed by position
    Position in chunks list matches position in FAISS index.
    """
    os.makedirs(output_dir, exist_ok=True)

    index_path = os.path.join(output_dir, f"{strategy}_index.faiss")
    chunks_path = os.path.join(output_dir, f"{strategy}_chunks.pkl")

    faiss.write_index(index, index_path)

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"saved index  → {index_path}")
    print(f"saved chunks → {chunks_path}")
    print(f"total vectors in index: {index.ntotal}")


def build_pipeline(jsonl_path: str, strategy: str):
    """
    Full pipeline for one strategy:
    1. Load filings from JSONL
    2. Chunk each filing using specified strategy
    3. Embed all chunks via OpenAI API
    4. Build FAISS index
    5. Save index and metadata to disk
    """
    print(f"\n{'='*60}")
    print(f"building {strategy} pipeline")
    print(f"model: {EMBEDDING_MODEL} ({EMBEDDING_DIM} dims)")
    print(f"filings: {DEV_FILINGS}")
    print(f"{'='*60}\n")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # load MiniLM for semantic chunking — not needed for fixed
    st_model = None
    if strategy == "semantic":
        print("loading MiniLM for semantic chunking...")
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("model loaded\n")

    all_chunks = []
    filing_count = 0

    print(f"chunking {DEV_FILINGS} filings...")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if filing_count >= DEV_FILINGS:
                break

            filing = json.loads(line.strip())
            filename = filing.get("filename", "unknown")

            if strategy == "fixed":
                chunks = chunk_filing_fixed(filing)
            else:
                chunks = chunk_filing_semantic(filing, st_model)

            all_chunks.extend(chunks)
            filing_count += 1
            print(f"  {filing_count}/{DEV_FILINGS} — {filename} — {len(chunks)} chunks")

    print(f"\ntotal chunks to embed: {len(all_chunks)}")

    # estimate cost before embedding
    # text-embedding-3-small costs $0.02 per million tokens
    # rough estimate: 1 chunk ≈ 200 tokens on average
    estimated_tokens = len(all_chunks) * 200
    estimated_cost = (estimated_tokens / 1_000_000) * 0.02
    print(f"estimated tokens: {estimated_tokens:,}")
    print(f"estimated cost: ${estimated_cost:.4f}")
    print("\nembedding chunks via OpenAI API...")

    embeddings = embed_chunks(all_chunks, client)

    print(f"\nembedding shape: {embeddings.shape}")
    print("building FAISS index...")

    index = build_faiss_index(embeddings)

    print("saving to disk...")
    save_index(index, all_chunks, strategy)

    print(f"\n{strategy} pipeline complete")
    return index, all_chunks


if __name__ == "__main__":
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--eloukas--edgar-corpus/"
        "snapshots/7e90f0f342569b35213445f809cfaf3b91f9964f/2018/train.jsonl"
    )

    if not os.path.exists(cache_path):
        print("cached file not found — run download.py first")
    else:
        # build fixed index first
        build_pipeline(cache_path, strategy="fixed")

        # then semantic
        build_pipeline(cache_path, strategy="semantic")