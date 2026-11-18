"""Dataset access for the AskThili AI Product Describer."""

import json
from pathlib import Path


CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "skincare_corpus.jsonl"


def load_product_corpus(path: str | Path | None = None) -> list[dict]:
    """Load product records from the bundled JSONL corpus or a supplied file."""
    corpus_path = Path(path) if path is not None else CORPUS_PATH
    records = []
    with corpus_path.open(encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {corpus_path} at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {corpus_path} at line {line_number}")
            records.append(record)
    return records


def milestone_1() -> list[dict]:
    """Return the curated product corpus."""
    return load_product_corpus()
