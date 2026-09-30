import json
import re
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "entities.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "normalized_entities.jsonl"


# ---------------------------------------------------------
# Safe normalization
# ---------------------------------------------------------

def normalize_entity(text):
    """
    Conservative normalization.

    We normalize formatting only.
    We do NOT attempt semantic merging.
    """

    text = text.strip().lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Normalize spaces around common punctuation
    text = re.sub(r"\s*&\s*", " & ", text)
    text = re.sub(r"\s*-\s*", "-", text)

    return text


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Reading:", INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}"
        )

    total_mentions = 0

    normalized_groups = defaultdict(list)

    # -----------------------------------------------------
    # Read entities
    # -----------------------------------------------------

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            original_text = record["entity_text"]
            entity_type = record["entity_type"]

            normalized_text = normalize_entity(original_text)

            total_mentions += 1

            normalized_groups[
                (normalized_text, entity_type)
            ].append(record)

    # -----------------------------------------------------
    # Create normalized records
    # -----------------------------------------------------

    output_records = []

    for (normalized_text, entity_type), mentions in normalized_groups.items():

        # Preserve all original mentions
        surface_forms = sorted(
            set(
                m["entity_text"]
                for m in mentions
            )
        )

        output_records.append({
            "canonical_text": normalized_text,
            "entity_type": entity_type,
            "mention_count": len(mentions),
            "surface_forms": surface_forms,
            "mentions": mentions
        })

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for record in output_records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("ENTITY NORMALIZATION COMPLETE")
    print("=" * 60)

    print(f"Original mentions       : {total_mentions}")
    print(f"Normalized entity nodes : {len(output_records)}")

    reduction = (
        1 - len(output_records) / total_mentions
    ) * 100

    print(f"Reduction               : {reduction:.2f}%")

    # -----------------------------------------------------
    # Show examples where multiple surface forms
    # map to the same normalized entity
    # -----------------------------------------------------

    print()
    print("Entities with multiple surface forms:")
    print("-" * 60)

    examples = 0

    for record in sorted(
        output_records,
        key=lambda x: x["mention_count"],
        reverse=True
    ):

        if len(record["surface_forms"]) > 1:

            print(
                f"{record['canonical_text']}"
                f" [{record['entity_type']}]"
            )

            print(
                f"  Mentions: {record['mention_count']}"
            )

            print(
                f"  Forms: {record['surface_forms']}"
            )

            print()

            examples += 1

            if examples >= 20:
                break

    print()
    print("Output:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()