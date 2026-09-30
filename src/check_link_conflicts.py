import json
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LINKING_FILE = (
    PROJECT_ROOT.parent
    / "anno-ctr-lrec-coling-2024"
    / "AnnoCTR"
    / "linking"
    / "train.jsonl"
)


def normalize(text):
    return " ".join(text.strip().lower().split())


groups = defaultdict(list)

with open(LINKING_FILE, "r", encoding="utf-8") as f:

    for line in f:

        if not line.strip():
            continue

        r = json.loads(line)

        key = (
            r["document"],
            normalize(r["mention"])
        )

        groups[key].append({
            "mention": r["mention"],
            "label_title": r.get("label_title"),
            "label_id": r.get("label_id"),
            "label_link": r.get("label_link"),
            "entity_type": r.get("entity_type")
        })


duplicate_keys = 0
conflicting_keys = 0

print("=" * 60)
print("ANNOCTR LINKING CONFLICT CHECK")
print("=" * 60)

for key, records in groups.items():

    if len(records) <= 1:
        continue

    duplicate_keys += 1

    canonical_ids = set(
        r["label_id"]
        for r in records
    )

    if len(canonical_ids) > 1:

        conflicting_keys += 1

        print()
        print("CONFLICT")
        print("Document:", key[0])
        print("Mention :", key[1])

        for r in records:
            print(
                f"  -> {r['label_title']} "
                f"(ID={r['label_id']}, "
                f"type={r['entity_type']})"
            )


print()
print("-" * 60)
print(f"Keys with duplicate annotations : {duplicate_keys}")
print(f"Keys with conflicting entities  : {conflicting_keys}")
print("-" * 60)