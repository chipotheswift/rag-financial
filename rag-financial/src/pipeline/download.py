import os
import json
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

SAMPLE_SIZE = 5

EXPECTED_SECTIONS = [
    "section_1", "section_1A", "section_1B", "section_2",
    "section_3", "section_4", "section_5", "section_6",
    "section_7", "section_7A", "section_8", "section_9",
    "section_9A", "section_9B", "section_10", "section_11",
    "section_12", "section_13", "section_14", "section_15",
]


def download_and_validate():
    print("downloading 2018/train.jsonl from EDGAR-CORPUS...")
    print("(one year subset — used for validation only)\n")

    # download just one year's training split directly as a file
    # jsonl = JSON Lines — one filing per line, easy to stream
    local_path = hf_hub_download(
        repo_id="eloukas/edgar-corpus",
        filename="2018/train.jsonl",
        repo_type="dataset",
    )
    print(f"file cached at: {local_path}\n")

    issues = []
    count = 0

    # open the file and read one line at a time
    # each line is one complete filing as a JSON object
    # this is streaming — we never load the whole file into memory
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            if count >= SAMPLE_SIZE:
                break

            # parse the line from JSON string into a Python dict
            filing = json.loads(line.strip())
            count += 1

            company = filing.get("company", "unknown")
            year = filing.get("year", "unknown")
            filename = filing.get("filename", "unknown")

            print(f"filing {count}: {company} - {year}")
            print(f"  filename : {filename}")

            missing = []
            empty = []

            for section in EXPECTED_SECTIONS:
                if section not in filing:
                    missing.append(section)
                elif not filing[section] or len(str(filing[section]).strip()) == 0:
                    empty.append(section)

            if missing:
                print(f"  MISSING sections : {missing}")
                issues.append((filename, "missing", missing))
            if empty:
                print(f"  empty sections   : {empty}")
            if not missing:
                print(f"  all 20 sections present ✓")

            # print a snippet of risk factors so we can see raw text
            risk = str(filing.get("section_1A", ""))
            snippet = risk[:300].replace("\n", " ").strip()
            print(f"  section_1A snippet: {snippet}...\n")

    print("=" * 60)
    if not issues:
        print(f"validation passed - all {count} filings look clean")
    else:
        print(f"validation found issues in {len(issues)} filings")

    print("\ndone. file cached locally by huggingface_hub.")


if __name__ == "__main__":
    download_and_validate()