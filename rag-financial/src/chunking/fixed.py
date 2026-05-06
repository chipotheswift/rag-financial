import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# these are the sections most likely to contain answerable content
# we skip sections that are typically empty or boilerplate
CONTENT_SECTIONS = [
    "section_1",   # Business Overview
    "section_1A",  # Risk Factors
    "section_7",   # MD&A
    "section_7A",  # Quantitative Disclosures
    "section_8",   # Financial Statements
]

# 512 tokens is the industry default for RAG pipelines
# 50 token overlap reduces information loss at chunk boundaries
# these are the baseline settings your proposal specifies
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def chunk_filing_fixed(filing: dict) -> list[dict]:
    """
    Takes one filing as a dict and returns a list of fixed-size chunks.
    Each chunk is a dict containing the text and metadata about where
    it came from — company, year, section, chunk index.
    Metadata is critical for your evaluation — you need to know which
    section a retrieved chunk came from to classify failure modes.
    """

    # RecursiveCharacterTextSplitter is LangChain's standard splitter
    # it tries to split on paragraphs first, then sentences, then words
    # falling back to characters only if necessary
    # this is slightly smarter than a pure character split but still
    # completely ignores semantic meaning
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,  # measures in characters not tokens
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    filename = filing.get("filename", "unknown")
    year = filing.get("year", "unknown")
    chunks = []

    for section in CONTENT_SECTIONS:
        text = filing.get(section, "")

        # skip empty or missing sections — filing 3 from our validation
        # had 11 empty sections, we don't want empty chunks in our index
        if not text or len(text.strip()) < 50:
            continue

        # split this section's text into chunks
        section_chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(section_chunks):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "filename": filename,
                    "year": year,
                    "section": section,
                    "chunk_index": i,
                    "total_chunks_in_section": len(section_chunks),
                    "strategy": "fixed",
                    "chunk_size_chars": len(chunk_text),
                }
            })

    return chunks


def preview_fixed_chunking(jsonl_path: str, num_filings: int = 2):
    """
    Loads a few filings from a JSONL file and prints chunk statistics
    so you can see exactly what fixed-size chunking produces.
    """
    print("=" * 60)
    print("FIXED-SIZE CHUNKING PREVIEW")
    print(f"chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print("=" * 60)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for filing_num, line in enumerate(f):
            if filing_num >= num_filings:
                break

            filing = json.loads(line.strip())
            filename = filing.get("filename", "unknown")
            chunks = chunk_filing_fixed(filing)

            print(f"\nfiling: {filename}")
            print(f"total chunks produced: {len(chunks)}")

            # show chunk counts per section
            from collections import Counter
            section_counts = Counter(c["metadata"]["section"] for c in chunks)
            for section, count in section_counts.items():
                print(f"  {section}: {count} chunks")

            # show first and last chunk of section_1A to illustrate
            # the boundary problem — watch how chunks start/end mid-thought
            section_1a_chunks = [
                c for c in chunks if c["metadata"]["section"] == "section_1A"
            ]

            if section_1a_chunks:
                print(f"\n  --- first chunk of section_1A ---")
                print(f"  {section_1a_chunks[0]['text'][:200]}...")
                print(f"\n  --- last chunk of section_1A ---")
                print(f"  ...{section_1a_chunks[-1]['text'][-200:]}")
                print(f"\n  --- chunk boundary example (chunks 1 and 2) ---")
                if len(section_1a_chunks) > 1:
                    print(f"  END OF CHUNK 1:")
                    print(f"  ...{section_1a_chunks[0]['text'][-100:]}")
                    print(f"  START OF CHUNK 2:")
                    print(f"  {section_1a_chunks[1]['text'][:100]}...")

            print("-" * 60)


if __name__ == "__main__":
    # path where huggingface_hub cached the file
    cache_path = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--eloukas--edgar-corpus/"
        "snapshots/7e90f0f342569b35213445f809cfaf3b91f9964f/2018/train.jsonl"
    )

    if not os.path.exists(cache_path):
        print("cached file not found — run download.py first")
    else:
        preview_fixed_chunking(cache_path, num_filings=2)