import json
from pathlib import Path


INPUT_PATH = Path("data/relation_random_audit_200.jsonl")
OUTPUT_PATH = Path("data/independent_annotation_1_200.jsonl")


def load_records(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_annotation_record(record):
    return {
        "audit_id": record["audit_id"],
        "document": record.get("document", ""),
        "segment": record.get("segment"),
        "source": record.get("source", ""),
        "source_type": record.get("source_type", ""),
        "relation": record.get("relation", ""),
        "target": record.get("target", ""),
        "evidence": record.get("sentence", ""),
        "human_label": "",
        "error_category": "",
        "annotator_notes": "",
    }


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    source_records = load_records(INPUT_PATH)
    if len(source_records) != 200:
        raise ValueError(f"Expected 200 source records, found {len(source_records)}")

    annotation_records = [make_annotation_record(record) for record in source_records]
    audit_ids = [int(record["audit_id"]) for record in annotation_records]
    if audit_ids != list(range(1, 201)):
        raise ValueError("Audit IDs must remain in original order from 1 through 200")

    if any(record["human_label"] for record in annotation_records):
        raise ValueError("Generated records contain non-blank human labels")
    if any(record["error_category"] for record in annotation_records):
        raise ValueError("Generated records contain non-blank error categories")
    if any(record["annotator_notes"] for record in annotation_records):
        raise ValueError("Generated records contain non-blank annotator notes")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for record in annotation_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    blank_labels = sum(not record["human_label"] for record in annotation_records)
    blank_categories = sum(not record["error_category"] for record in annotation_records)
    blank_notes = sum(not record["annotator_notes"] for record in annotation_records)

    print(f"Records: {len(annotation_records)}")
    print(f"Minimum audit ID: {min(audit_ids)}")
    print(f"Maximum audit ID: {max(audit_ids)}")
    print(f"Blank human labels: {blank_labels}")
    print(f"Blank error categories: {blank_categories}")
    print(f"Blank notes: {blank_notes}")
    print("Independent annotation set prepared.")
    print("No previous human labels were copied.")
    print("Annotate these examples independently before evaluation.")


if __name__ == "__main__":
    main()