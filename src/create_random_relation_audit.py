import json
import random
from pathlib import Path

INPUT_PATH = Path("data/annoctr_relation_predictions_arguments_v2.jsonl")
OUTPUT_PATH = Path("data/relation_random_audit_200.jsonl")

SAMPLE_SIZE = 200
SEED = 42


def main():

    records = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Total predictions available: {len(records)}")

    if len(records) < SAMPLE_SIZE:
        raise ValueError(
            f"Need {SAMPLE_SIZE} records, "
            f"but only {len(records)} available."
        )

    random.seed(SEED)

    sampled = random.sample(
        records,
        SAMPLE_SIZE
    )

    # Give each sampled record a stable audit ID.
    audit_records = []

    for i, record in enumerate(sampled, start=1):

        audit_records.append({
            "audit_id": i,

            "source": record.get(
                "source",
                record.get("source_mention", "")
            ),

            "source_type": record.get(
                "source_type",
                ""
            ),

            "relation": record.get(
                "relation",
                record.get("predicted_relation", "")
            ),

            "target": record.get(
                "target",
                record.get("target_mention", "")
            ),

            "target_type": record.get(
                "target_type",
                ""
            ),

            "svm_margin": record.get(
                "margin",
                record.get("svm_margin", 0)
            ),

            "segment": record.get(
                "segment"
            ),

            "document": record.get(
                "document"
            ),

            "sentence": record.get(
                "sentence",
                record.get("text", "")
            ),

            # To be filled manually during audit.
            "human_label": "",

            "error_category": ""
        })

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:

        for record in audit_records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    print("=" * 60)
    print("RANDOM RELATION AUDIT CREATED")
    print("=" * 60)
    print(f"Sample size : {len(audit_records)}")
    print(f"Random seed : {SEED}")
    print(f"Output      : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()