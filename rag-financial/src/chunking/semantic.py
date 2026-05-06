import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CONTENT_SECTIONS = [
    "section_1",
    "section_1A",
    "section_7",
    "section_7A",
    "section_8",
]

# similarity threshold — when cosine similarity between adjacent
# sentences drops below this value we cut a new chunk there
# 0.5 is a good starting point — tune this during evaluation
SIMILARITY_THRESHOLD = 0.5

# even if similarity stays high we don't want chunks longer than this
# prevents one giant chunk if a topic is discussed for many paragraphs
MAX_CHUNK_CHARS = 2000

# minimum chunk size — avoids creating chunks that are just one
# sentence of boilerplate like "Item 1A. Risk Factors."
MIN_CHUNK_CHARS = 100


def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using basic punctuation rules.
    We use a simple approach here — pysbd (installed with ragas)
    would be more accurate but adds complexity we don't need yet.
    """
    import re

    # split on period/exclamation/question followed by space and capital
    # this handles most financial text sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # filter out empty strings and very short fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Measures the angle between two vectors.
    Returns 1.0 for identical meaning, 0.0 for completely unrelated.
    This is the core math behind semantic search.
    """
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    # guard against division by zero if a vector is all zeros
    if magnitude == 0:
        return 0.0
    return dot_product / magnitude


def chunk_section_semantic(
    text: str,
    model: SentenceTransformer,
    section: str,
    filename: str,
    year: str,
) -> list[dict]:
    """
    Takes one section's text and returns semantically coherent chunks.
    Each chunk groups sentences that are talking about the same topic.
    """

    sentences = split_into_sentences(text)

    if not sentences:
        return []

    # embed all sentences at once — batching is much faster than
    # embedding one sentence at a time
    # show_progress_bar=False keeps output clean
    embeddings = model.encode(sentences, show_progress_bar=False)

    chunks = []
    current_sentences = [sentences[0]]
    current_chars = len(sentences[0])

    for i in range(1, len(sentences)):
        # measure similarity between current sentence and previous
        similarity = cosine_similarity(embeddings[i], embeddings[i - 1])

        # decide whether to continue current chunk or start a new one
        # three conditions that trigger a new chunk:
        # 1. similarity drops below threshold — topic changed
        # 2. current chunk is getting too long — enforce max size
        # 3. always at least one sentence per chunk — enforce min
        topic_shifted = similarity < SIMILARITY_THRESHOLD
        too_long = current_chars + len(sentences[i]) > MAX_CHUNK_CHARS

        if (topic_shifted or too_long) and current_chars >= MIN_CHUNK_CHARS:
            # save the current chunk
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "filename": filename,
                    "year": year,
                    "section": section,
                    "chunk_index": len(chunks),
                    "strategy": "semantic",
                    "chunk_size_chars": len(chunk_text),
                    "similarity_at_boundary": round(similarity, 4),
                }
            })

            # start a new chunk with current sentence
            current_sentences = [sentences[i]]
            current_chars = len(sentences[i])
        else:
            # continue building current chunk
            current_sentences.append(sentences[i])
            current_chars += len(sentences[i])

    # don't forget the last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        if len(chunk_text.strip()) >= MIN_CHUNK_CHARS:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "filename": filename,
                    "year": year,
                    "section": section,
                    "chunk_index": len(chunks),
                    "strategy": "semantic",
                    "chunk_size_chars": len(chunk_text),
                    "similarity_at_boundary": None,
                }
            })

    return chunks


def chunk_filing_semantic(
    filing: dict,
    model: SentenceTransformer
) -> list[dict]:
    """
    Runs semantic chunking on all content sections of one filing.
    """
    filename = filing.get("filename", "unknown")
    year = filing.get("year", "unknown")
    all_chunks = []

    for section in CONTENT_SECTIONS:
        text = filing.get(section, "")

        if not text or len(text.strip()) < MIN_CHUNK_CHARS:
            continue

        section_chunks = chunk_section_semantic(
            text, model, section, filename, year
        )
        all_chunks.extend(section_chunks)

    return all_chunks


def preview_semantic_chunking(jsonl_path: str, num_filings: int = 2):
    """
    Loads a few filings and prints semantic chunk statistics.
    Run this alongside fixed.py output to see the difference.
    """
    print("=" * 60)
    print("SEMANTIC CHUNKING PREVIEW")
    print(f"similarity_threshold={SIMILARITY_THRESHOLD}")
    print("loading MiniLM model...")

    # load the model — it's already cached in the Docker image
    # from the RUN command in the Dockerfile
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("model loaded\n")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for filing_num, line in enumerate(f):
            if filing_num >= num_filings:
                break

            filing = json.loads(line.strip())
            filename = filing.get("filename", "unknown")

            print(f"chunking: {filename}")
            chunks = chunk_filing_semantic(filing, model)

            print(f"total chunks produced: {len(chunks)}")

            from collections import Counter
            section_counts = Counter(
                c["metadata"]["section"] for c in chunks
            )
            for section, count in section_counts.items():
                print(f"  {section}: {count} chunks")

            # show section_1A chunks and their boundaries
            section_1a_chunks = [
                c for c in chunks
                if c["metadata"]["section"] == "section_1A"
            ]

            if section_1a_chunks:
                print(f"\n  --- first chunk of section_1A ---")
                print(f"  {section_1a_chunks[0]['text'][:200]}...")
                print(f"\n  --- last chunk of section_1A ---")
                print(f"  ...{section_1a_chunks[-1]['text'][-200:]}")
                print(f"\n  --- chunk boundary example (chunks 1 and 2) ---")
                if len(section_1a_chunks) > 1:
                    print(f"  END OF CHUNK 1:")
                    print(f"  ...{section_1a_chunks[0]['text'][-150:]}")
                    print(f"  similarity at boundary: "
                          f"{section_1a_chunks[1]['metadata']['similarity_at_boundary']}")
                    print(f"  START OF CHUNK 2:")
                    print(f"  {section_1a_chunks[1]['text'][:150]}...")

            print("-" * 60)


if __name__ == "__main__":
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--eloukas--edgar-corpus/"
        "snapshots/7e90f0f342569b35213445f809cfaf3b91f9964f/2018/train.jsonl"
    )

    if not os.path.exists(cache_path):
        print("cached file not found — run download.py first")
    else:
        preview_semantic_chunking(cache_path, num_filings=2)