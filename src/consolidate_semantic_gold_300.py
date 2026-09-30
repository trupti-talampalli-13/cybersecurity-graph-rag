import json
from collections import Counter
from pathlib import Path


AUDIT_200 = Path("data/relation_random_audit_200.jsonl")
PROVISIONAL_100 = Path("data/semantic_annotation_201_300_provisional.jsonl")
OUTPUT = Path("data/semantic_gold_300.jsonl")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_audit_record(record):
    normalized = dict(record)
    normalized["gold_id"] = int(record["audit_id"])
    normalized["sentence"] = record.get("sentence", "")
    normalized.pop("audit_id", None)
    return normalized


def validate_records(records, expected_ids, name):
    actual_ids = [int(record["gold_id"]) for record in records]
    if sorted(actual_ids) != expected_ids:
        raise ValueError(
            f"{name} must contain exactly IDs {expected_ids[0]}-{expected_ids[-1]}"
        )

    missing_labels = [
        record["gold_id"]
        for record in records
        if record.get("human_label") not in {"TRUE", "FALSE", "UNCERTAIN"}
    ]
    if missing_labels:
        preview = ", ".join(str(value) for value in missing_labels[:10])
        raise ValueError(
            f"{name} has {len(missing_labels)} missing or invalid human labels "
            f"(examples: {preview}). Complete independent annotations before "
            "creating the frozen gold set."
        )


def main():
    for path in (AUDIT_200, PROVISIONAL_100):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    audit_records = [normalize_audit_record(record) for record in load_jsonl(AUDIT_200)]
    provisional_records = load_jsonl(PROVISIONAL_100)

    validate_records(audit_records, list(range(1, 201)), "1-200 audit")
    validate_records(provisional_records, list(range(201, 301)), "201-300 annotations")

    records = sorted(audit_records + provisional_records, key=lambda item: item["gold_id"])
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    label_counts = Counter(record["human_label"] for record in records)
    print(f"Created {OUTPUT} with {len(records)} records")
    print("Human labels:")
    for label in ("TRUE", "FALSE", "UNCERTAIN"):
        print(f"  {label:10s}: {label_counts[label]}")


if __name__ == "__main__":
    main()