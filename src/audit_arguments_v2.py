import json
from collections import Counter, defaultdict

INPUT_FILE = "data/annoctr_relation_predictions_arguments_v2.jsonl"

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print("=" * 80)
print("ARGUMENT EXTRACTION V2 AUDIT")
print("=" * 80)

print(f"Total records: {len(records)}")

# ------------------------------------------------------------
# Counters
# ------------------------------------------------------------

structure_counts = Counter()
relation_structure = Counter()

for r in records:
    a = r["relation_arguments_v2"]

    structure = a.get("argument_structure", "none")
    relation = r["predicted_relation"]

    structure_counts[structure] += 1
    relation_structure[(relation, structure)] += 1

print("\nArgument structures:")
for k, v in structure_counts.most_common():
    print(f"  {k:50s} {v}")

# ------------------------------------------------------------
# Print resolved examples
# ------------------------------------------------------------

resolved = [
    r for r in records
    if r["relation_arguments_v2"].get("direction") != "unknown"
]

resolved.sort(
    key=lambda r: r.get("margin", 0),
    reverse=True
)

print("\n" + "=" * 80)
print("TOP RESOLVED EXAMPLES")
print("=" * 80)

for i, r in enumerate(resolved[:50], 1):

    a = r["relation_arguments_v2"]

    print(f"\n[{i}]")
    print(
        f"{r['source_mention']} "
        f"--[{r['predicted_relation']}]--> "
        f"{r['target_mention']}"
    )

    print(f"Margin      : {r.get('margin', 0):.4f}")
    print(f"Direction   : {a.get('direction')}")
    print(f"Structure   : {a.get('argument_structure')}")
    print(f"Source role : {a.get('source_role')}")
    print(f"Target role : {a.get('target_role')}")
    print(f"Trigger     : {a.get('trigger')}")
    print(f"Sentence    : {a.get('sentence')}")

# ------------------------------------------------------------
# Print reverse examples
# ------------------------------------------------------------

reverse = [
    r for r in records
    if r["relation_arguments_v2"].get("direction") == "reverse"
]

reverse.sort(
    key=lambda r: r.get("margin", 0),
    reverse=True
)

print("\n" + "=" * 80)
print("REVERSE-DIRECTION EXAMPLES")
print("=" * 80)

for i, r in enumerate(reverse[:30], 1):

    a = r["relation_arguments_v2"]

    print(f"\n[{i}]")
    print(
        f"SVM: {r['source_mention']} "
        f"--[{r['predicted_relation']}]--> "
        f"{r['target_mention']}"
    )

    print(f"Margin    : {r.get('margin', 0):.4f}")
    print(f"Structure : {a.get('argument_structure')}")
    print(f"Trigger   : {a.get('trigger')}")
    print(f"Sentence  : {a.get('sentence')}")

# ------------------------------------------------------------
# High-margin unknowns
# ------------------------------------------------------------

unknown = [
    r for r in records
    if r["relation_arguments_v2"].get("direction") == "unknown"
]

unknown.sort(
    key=lambda r: r.get("margin", 0),
    reverse=True
)

print("\n" + "=" * 80)
print("HIGH-MARGIN UNKNOWN EXAMPLES")
print("=" * 80)

for i, r in enumerate(unknown[:50], 1):

    a = r["relation_arguments_v2"]

    print(f"\n[{i}]")
    print(
        f"{r['source_mention']} "
        f"--[{r['predicted_relation']}]--> "
        f"{r['target_mention']}"
    )

    print(f"Margin  : {r.get('margin', 0):.4f}")
    print(f"Trigger : {a.get('trigger')}")
    print(f"Sentence: {a.get('sentence')}")

# ------------------------------------------------------------
# Relation-level resolved rates
# ------------------------------------------------------------

relation_total = Counter()
relation_resolved = Counter()

for r in records:

    relation = r["predicted_relation"]

    relation_total[relation] += 1

    if r["relation_arguments_v2"].get("direction") != "unknown":
        relation_resolved[relation] += 1

print("\n" + "=" * 80)
print("RESOLUTION RATE BY RELATION")
print("=" * 80)

for relation, total in relation_total.most_common():

    resolved_count = relation_resolved[relation]

    rate = resolved_count / total if total else 0

    print(
        f"{relation:25s} "
        f"{resolved_count:4d}/{total:4d} "
        f"({rate:.2%})"
    )