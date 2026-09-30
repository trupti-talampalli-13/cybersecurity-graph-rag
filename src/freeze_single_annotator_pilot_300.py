import json
from collections import Counter
from pathlib import Path


PRIMARY_PATH = Path("data/independent_annotation_1_200.jsonl")
PROVISIONAL_PATH = Path("data/semantic_annotation_201_300_provisional.jsonl")
OUTPUT_PATH = Path("data/semantic_pilot_300_frozen.jsonl")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_primary(record):
    return {
        "pilot_id": int(record["audit_id"]),
        "source": record.get("source", ""),
        "source_type": record.get("source_type", ""),
        "relation": record.get("relation", ""),
        "target": record.get("target", ""),
        "target_type": record.get("target_type", ""),
        "evidence_text": record.get("evidence", ""),
        "document": record.get("document", ""),
        "segment": record.get("segment"),
        "human_label": record["human_label"],
        "error_category": record.get("error_category", ""),
        "annotation_provenance": "single_annotator_manual_pilot",
        "annotator_notes": record.get("annotator_notes", ""),
    }


def normalize_provisional(record):
    return {
        "pilot_id": int(record["gold_id"]),
        "source": record.get("source", ""),
        "source_type": record.get("source_type", ""),
        "relation": record.get("relation", ""),
        "target": record.get("target", ""),
        "target_type": record.get("target_type", ""),
        "evidence_text": record.get("sentence", ""),
        "document": record.get("document", ""),
        "segment": record.get("segment"),
        "human_label": record["human_label"],
        "error_category": record.get("error_category", ""),
        "annotation_provenance": "provisional_assistant_assisted",
        "annotator_notes": record.get("annotator_notes", ""),
    }


def main():
    primary = load_jsonl(PRIMARY_PATH)
    provisional = load_jsonl(PROVISIONAL_PATH)
    if len(primary) != 200 or len(provisional) != 100:
        raise ValueError("Expected 200 primary and 100 provisional records")

    records = [normalize_primary(record) for record in primary]
    records.extend(normalize_provisional(record) for record in provisional)
    records.sort(key=lambda record: record["pilot_id"])

    ids = [record["pilot_id"] for record in records]
    if ids != list(range(1, 301)):
        raise ValueError("Pilot IDs must be unique and cover exactly 1-300")
    allowed = {"TRUE", "FALSE", "UNCERTAIN"}
    if any(record["human_label"] not in allowed for record in records):
        raise ValueError("Every pilot record must have TRUE, FALSE, or UNCERTAIN")

    OUTPUT_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    print(f"Frozen pilot records: {len(records)}")
    print(f"Label distribution: {dict(Counter(record['human_label'] for record in records))}")
    print("Provenance:", dict(Counter(record["annotation_provenance"] for record in records)))
    print(f"Output: {OUTPUT_PATH}")
    print("Evaluation type: single_annotator_pilot")
    print("Inter-annotator agreement: unavailable")
    print("Pilot set frozen without threshold tuning.")


if __name__ == "__main__":
    main()