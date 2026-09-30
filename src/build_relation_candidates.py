import json
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

SEGMENTS_FILE = BASE_DIR / "data" / "train_segments.jsonl"
ENTITIES_FILE = BASE_DIR / "data" / "entities.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "annoctr_relation_candidates.jsonl"


# ---------------------------------------------------------
# Load AnnoCTR segments
# ---------------------------------------------------------

def load_segments(path):
    segments = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            key = (
                record["document"],
                record["segment"]
            )

            segments[key] = record["text"]

    return segments


# ---------------------------------------------------------
# Load entity mentions
# ---------------------------------------------------------

def load_entities(path):
    entities = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            key = (
                record["document"],
                record["segment"]
            )

            entities.setdefault(key, []).append({
                "text": record["entity_text"],
                "type": record["entity_type"]
            })

    return entities


# ---------------------------------------------------------
# Generate ordered entity pairs
# ---------------------------------------------------------

def generate_candidates(segments, entities):

    candidates = []

    pair_count = 0
    skipped_segments = 0

    for key, text in segments.items():

        mentions = entities.get(key, [])

        # Need at least two entities for a relation candidate
        if len(mentions) < 2:
            skipped_segments += 1
            continue

        # -------------------------------------------------
        # Remove exact duplicate mentions within a segment
        # -------------------------------------------------

        unique_mentions = []
        seen = set()

        for entity in mentions:

            mention_key = (
                entity["text"].strip().lower(),
                entity["type"]
            )

            if mention_key not in seen:
                seen.add(mention_key)
                unique_mentions.append(entity)

        # -------------------------------------------------
        # Generate ordered pairs
        # -------------------------------------------------

        for i, source in enumerate(unique_mentions):

            for j, target in enumerate(unique_mentions):

                if i == j:
                    continue

                # Avoid identical surface-form/type pairs
                if (
                    source["text"].strip().lower()
                    == target["text"].strip().lower()
                    and source["type"] == target["type"]
                ):
                    continue

                candidate = {
                    "document": key[0],
                    "segment": key[1],
                    "text": text,
                    "source_mention": source["text"],
                    "source_type": source["type"],
                    "target_mention": target["text"],
                    "target_type": target["type"]
                }

                candidates.append(candidate)
                pair_count += 1

    return candidates, skipped_segments


# ---------------------------------------------------------
# Save candidates
# ---------------------------------------------------------

def save_candidates(candidates, path):

    with open(path, "w", encoding="utf-8") as f:

        for candidate in candidates:
            f.write(
                json.dumps(
                    candidate,
                    ensure_ascii=False
                ) + "\n"
            )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("AnnoCTR Relation Candidate Generation")
    print("=" * 60)

    print("\nLoading segments...")
    segments = load_segments(SEGMENTS_FILE)

    print(f"Segments loaded: {len(segments)}")

    print("\nLoading entity mentions...")
    entities = load_entities(ENTITIES_FILE)

    print(f"Segments containing entities: {len(entities)}")

    print("\nGenerating ordered entity pairs...")

    candidates, skipped = generate_candidates(
        segments,
        entities
    )

    print(f"Candidate pairs generated: {len(candidates)}")
    print(f"Segments skipped (<2 entities): {skipped}")

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    type_pairs = Counter()

    for candidate in candidates:

        pair = (
            candidate["source_type"],
            candidate["target_type"]
        )

        type_pairs[pair] += 1

    print("\nTop entity-type pair combinations:")

    for (source_type, target_type), count in type_pairs.most_common(20):

        print(
            f"{source_type:25s} -> "
            f"{target_type:25s} : {count}"
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_candidates(
        candidates,
        OUTPUT_FILE
    )

    print("\nSaved:")
    print(OUTPUT_FILE)

    # -----------------------------------------------------
    # Show examples
    # -----------------------------------------------------

    print("\nExample candidates:")
    print("-" * 60)

    for candidate in candidates[:10]:

        print(
            f"\n[{candidate['source_type']}] "
            f"{candidate['source_mention']}"
        )

        print(
            f"    -> [{candidate['target_type']}] "
            f"{candidate['target_mention']}"
        )

        print(
            f"Document: {candidate['document']}"
        )

        print(
            f"Segment: {candidate['segment']}"
        )


if __name__ == "__main__":
    main()