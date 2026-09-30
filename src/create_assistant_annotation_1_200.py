import json
from collections import Counter
from pathlib import Path


INPUT_PATH = Path("data/independent_annotation_1_200.jsonl")
OUTPUT_PATH = Path("data/assistant_annotation_1_200.jsonl")

TRUE_IDS = {14, 125, 127, 160, 179}
UNCERTAIN_IDS = {56, 88, 95, 97}


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    records = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 200:
        raise ValueError(f"Expected 200 records, found {len(records)}")

    output = []
    for record in records:
        item = dict(record)
        audit_id = int(item["audit_id"])
        if audit_id in TRUE_IDS:
            item["human_label"] = "TRUE"
            item["error_category"] = "VALID_RELATION"
        elif audit_id in UNCERTAIN_IDS:
            item["human_label"] = "UNCERTAIN"
            item["error_category"] = "OTHER"
        else:
            item["human_label"] = "FALSE"
            item["error_category"] = "OTHER"
        item["annotator_notes"] = (
            "Assistant-assisted provisional annotation based on the declared pilot labels; "
            "not independent human annotation and not final gold data."
        )
        output.append(item)

    OUTPUT_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in output),
        encoding="utf-8",
    )

    print(f"Records: {len(output)}")
    print(f"Label distribution: {dict(Counter(record['human_label'] for record in output))}")
    print(f"Output: {OUTPUT_PATH}")
    print("Assistant-assisted annotation created.")
    print("This file is not independent human gold data.")


if __name__ == "__main__":
    main()