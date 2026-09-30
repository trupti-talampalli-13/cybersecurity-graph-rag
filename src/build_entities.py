import json
from pathlib import Path
from collections import Counter


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "train_segments.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "entities.jsonl"


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Reading:", INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}"
        )

    total_segments = 0
    total_entities = 0

    entity_type_counter = Counter()
    entity_text_counter = Counter()

    output_records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            document = record["document"]
            segment = record["segment"]
            entities = record.get("entities", [])

            total_segments += 1

            for entity in entities:

                entity_text = entity["text"]
                entity_type = entity["type"]

                total_entities += 1

                entity_type_counter[entity_type] += 1
                entity_text_counter[
                    entity_text.lower()
                ] += 1

                output_records.append({
                    "document": document,
                    "segment": segment,
                    "entity_text": entity_text,
                    "entity_type": entity_type
                })

    # -----------------------------------------------------
    # Save clean entity table
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
    print("ENTITY EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"Segments processed : {total_segments}")
    print(f"Entities extracted : {total_entities}")
    print(f"Unique entity texts: {len(entity_text_counter)}")

    print()
    print("Entity counts by type:")
    print("-" * 40)

    for entity_type, count in entity_type_counter.most_common():
        print(f"{entity_type:20s} {count}")

    print()
    print("Top 20 entity mentions:")
    print("-" * 40)

    for entity, count in entity_text_counter.most_common(20):
        print(f"{entity:35s} {count}")

    print()
    print("Output:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()