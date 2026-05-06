import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# how many filings to sample for validation
SAMPLE_SIZE = 5

# these are the 20 structured sections EDGAR-CORPUS splits every 10-K into
EXPECTED_SECTIONS = [
    "section_1",   # Business Overview
    "section_1A",  # Risk Factors
    "section_1B",  # Unresolved Staff Comments
    "section_2",   # Properties
    "section_3",   # Legal Proceedings
    "section_4",   # Mine Safety
    "section_5",   # Market for Registrant
    "section_6",   # Selected Financial Data
    "section_7",   # MD&A
    "section_7A",  # Quantitative Disclosures
    "section_8",   # Financial Statements
    "section_9",   # Changes in Accountants
    "section_9A",  # Controls and Procedures
    "section_9B",  # Other Information
    "section_10",  # Directors and Officers
    "section_11",  # Executive Compensation
    "section_12",  # Security Ownership
    "section_13",  # Certain Relationships
    "section_14",  # Principal Accountant Fees
    "section_15",  # Exhibits
]


def download_and_validate():
    print("connecting to EDGAR-CORPUS via streaming...")
    print("(no data downloaded yet — stream opens on first iteration)\n")

    # streaming=True is the key — nothing hits disk until we iterate
    dataset = load_dataset(
        "eloukas/edgar-corpus",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    print(f"sampling {SAMPLE_SIZE} filings for validation...\n")

    issues = []

    # dataset is now a generator — each call to next() fetches one filing
    for i, filing in enumerate(dataset):
        if i >= SAMPLE_SIZE:
            break

        # pull identifying info
        company = filing.get("company", "unknown")
        year = filing.get("year", "unknown")
        filename = filing.get("filename", "unknown")

        print(f"filing {i+1}: {company} — {year}")
        print(f"  filename : {filename}")

        # check every expected section exists and has content
        missing = []
        empty = []

        for section in EXPECTED_SECTIONS:
            if section not in filing:
                missing.append(section)
            elif not filing[section] or len(filing[section].strip()) == 0:
                empty.append(section)

        if missing:
            print(f"  MISSING sections : {missing}")
            issues.append((filename, "missing", missing))
        if empty:
            # empty sections are common — some companies skip optional sections
            print(f"  empty sections   : {empty}")
        if not missing:
            print(f"  all 20 sections present ✓")

        # show a snippet of Risk Factors so you can see what raw text looks like
        risk = filing.get("section_1A", "")
        snippet = risk[:300].replace("\n", " ").strip()
        print(f"  section_1A snippet: {snippet}...\n")

    # summary
    print("=" * 60)
    if not issues:
        print(f"validation passed — all {SAMPLE_SIZE} filings look clean")
        print("EDGAR-CORPUS is loading correctly via stream")
    else:
        print(f"validation found issues in {len(issues)} filings — check above")

    print("\ndone. no data written to disk — this was stream-only.")


if __name__ == "__main__":
    download_and_validate()