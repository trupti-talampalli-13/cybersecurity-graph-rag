import json
import os


# =========================================================
# CONFIG
# =========================================================

INPUT_FILE = "data/semantic_gold_additional_100.jsonl"

OUTPUT_FILE = "data/semantic_annotation_201_300.jsonl"


# =========================================================
# LOAD RAW GOLD SAMPLE
# =========================================================

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        records.append(json.loads(line))


print(f"Loaded {len(records)} raw examples")


# =========================================================
# VALIDATE GOLD IDS
# =========================================================

gold_ids = [
    r.get("gold_id")
    for r in records
]

expected_ids = list(range(201, 301))

if sorted(gold_ids) != expected_ids:

    raise ValueError(
        "Gold IDs are not exactly 201–300."
    )


# =========================================================
# CREATE ANNOTATION FILE
# =========================================================

output = []

for record in records:

    item = {

        # -------------------------------------------------
        # Identity
        # -------------------------------------------------

        "gold_id": record["gold_id"],

        # -------------------------------------------------
        # Candidate relation
        # -------------------------------------------------

        "source": record.get(
            "source",
            ""
        ),

        "source_type": record.get(
            "source_type",
            ""
        ),

        "relation": record.get(
            "relation",
            ""
        ),

        "target": record.get(
            "target",
            ""
        ),

        "target_type": record.get(
            "target_type",
            ""
        ),

        # -------------------------------------------------
        # Evidence
        # -------------------------------------------------

        "sentence": record.get(
            "sentence",
            ""
        ),

        # -------------------------------------------------
        # Model information
        # -------------------------------------------------

        "v6_status": record.get(
            "v6_status",
            ""
        ),

        "predicate_status": record.get(
            "predicate_status",
            ""
        ),

        "svm_margin": record.get(
            "svm_margin",
            None
        ),

        # -------------------------------------------------
        # Provenance
        # -------------------------------------------------

        "document": record.get(
            "document",
            ""
        ),

        "segment": record.get(
            "segment",
            None
        ),

        # -------------------------------------------------
        # HUMAN ANNOTATION
        #
        # Leave these blank initially.
        # -------------------------------------------------

        "human_label": "",

        "error_category": "",

        "annotator_notes": ""
    }

    output.append(item)


# =========================================================
# SAVE
# =========================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for item in output:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            ) + "\n"
        )


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 60)
print("SEMANTIC ANNOTATION FILE CREATED")
print("=" * 60)

print(
    f"Examples : {len(output)}"
)

print(
    f"IDs      : {output[0]['gold_id']}–"
    f"{output[-1]['gold_id']}"
)

print(
    f"Output   : {OUTPUT_FILE}"
)

print()
print("Annotation fields:")
print("  human_label")
print("  error_category")
print("  annotator_notes")

print()
print("Allowed human_label values:")
print("  TRUE")
print("  FALSE")
print("  UNCERTAIN")

print()
print("Allowed error_category values:")
print("  VALID_RELATION")
print("  WRONG_SUBJECT")
print("  WRONG_OBJECT")
print("  WRONG_DIRECTION")
print("  COORDINATION_ERROR")
print("  NESTED_ENTITY")
print("  REPORTING_CONTEXT")
print("  COMPARISON")
print("  TABLE")
print("  SEMANTIC_MISMATCH")
print("  OTHER")

print()
print("=" * 60)