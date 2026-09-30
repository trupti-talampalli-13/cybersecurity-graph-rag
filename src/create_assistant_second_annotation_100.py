import json
from collections import Counter
from pathlib import Path


SECOND_INPUT = Path("data/second_annotator_100.jsonl")
PRIMARY_INPUT = Path("data/independent_annotation_1_200.jsonl")
OUTPUT = Path("data/assistant_second_annotation_100.jsonl")


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    second_records = load_jsonl(SECOND_INPUT)
    primary_by_id = {
        int(record["audit_id"]): record for record in load_jsonl(PRIMARY_INPUT)
    }
    if len(second_records) != 100:
        raise ValueError(f"Expected 100 second-annotation records, found {len(second_records)}")

    output = []
    for record in second_records:
        audit_id = int(record["audit_id"])
        primary = primary_by_id.get(audit_id)
        if primary is None or not primary.get("human_label"):
            raise ValueError(f"Missing primary annotation for audit_id {audit_id}")
        item = dict(record)
        item["second_human_label"] = primary["human_label"]
        item["second_error_category"] = primary.get("error_category", "OTHER")
        item["second_annotator_notes"] = (
            "Assistant-assisted provisional annotation copied from the primary labels; "
            "not an independent second annotation and invalid for agreement estimation."
        )
        output.append(item)

    OUTPUT.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in output),
        encoding="utf-8",
    )
    print(f"Records: {len(output)}")
    print(f"Labels: {dict(Counter(record['second_human_label'] for record in output))}")
    print(f"Output: {OUTPUT}")
    print("Assistant-assisted second annotation created.")
    print("Do not use this file for Cohen's kappa or independent agreement claims.")


if __name__ == "__main__":
    main()