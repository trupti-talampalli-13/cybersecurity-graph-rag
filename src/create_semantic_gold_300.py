import json
import random
from collections import defaultdict


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "data/annoctr_relation_predictions_validated_v6.jsonl"
AUDIT_FILE = "data/relation_random_audit_200.jsonl"
OUTPUT_FILE = "data/semantic_gold_additional_100.jsonl"

SEED = 42

random.seed(SEED)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def make_key(record):
    """
    Create a unique key for a relation prediction.

    We use document + segment + source + relation + target
    so that examples already present in the 200-example
    random audit are not sampled again.
    """

    source = record.get(
        "source",
        record.get("source_mention", "")
    )

    relation = record.get(
        "relation",
        record.get("predicted_relation", "")
    )

    target = record.get(
        "target",
        record.get("target_mention", "")
    )

    return (
        record.get("document", ""),
        record.get("segment"),
        source,
        relation,
        target
    )


def sample_status(by_status, status, n):
    """
    Randomly sample n examples from a particular V6 status.
    """

    pool = by_status.get(status, [])

    if len(pool) < n:
        raise ValueError(
            f"Not enough examples for {status}: "
            f"needed {n}, available {len(pool)}"
        )

    return random.sample(pool, n)


# =========================================================
# STEP 1 — LOAD V6 PREDICTIONS
# =========================================================

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        records.append(json.loads(line))


print(f"Loaded {len(records)} V6 predictions")


# =========================================================
# STEP 2 — LOAD EXISTING 200-EXAMPLE AUDIT
# =========================================================

audited_keys = set()

with open(AUDIT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        key = make_key(record)

        audited_keys.add(key)


print(
    f"Existing audited candidates: "
    f"{len(audited_keys)}"
)


# =========================================================
# STEP 3 — REMOVE ALREADY AUDITED EXAMPLES
# =========================================================

remaining = []

for record in records:

    key = make_key(record)

    if key not in audited_keys:

        remaining.append(record)


print(
    f"Remaining candidates: "
    f"{len(remaining)}"
)


# =========================================================
# STEP 4 — GROUP BY V6 STATUS
# =========================================================

by_status = defaultdict(list)

for record in remaining:

    status = record.get(
        "status",
        "UNKNOWN"
    )

    by_status[status].append(record)


print()

print("Available V6 status distribution:")

for status in [
    "SUPPORTED",
    "PLAUSIBLE",
    "UNSUPPORTED",
    "UNVERIFIED",
    "UNKNOWN"
]:

    print(
        f"  {status:12s}: "
        f"{len(by_status.get(status, []))}"
    )


# =========================================================
# STEP 5 — BUILD THE 100-EXAMPLE STRATIFIED SET
# =========================================================
#
# Current V6 distribution:
#
# SUPPORTED   = 4
# PLAUSIBLE   = 875
# UNSUPPORTED = 238
# UNVERIFIED  = 4217
#
# We cannot sample 10 SUPPORTED examples because only
# 4 exist.
#
# Therefore:
#
# SUPPORTED   = ALL 4
# PLAUSIBLE   = 30
# UNSUPPORTED = 20
# UNVERIFIED  = 46
#
# Total = 100
#
# =========================================================

selected = []


# ---------------------------------------------------------
# SUPPORTED
# ---------------------------------------------------------

supported_pool = by_status.get(
    "SUPPORTED",
    []
)

print()
print(
    f"Using all {len(supported_pool)} "
    f"SUPPORTED examples"
)

selected.extend(supported_pool)


# ---------------------------------------------------------
# PLAUSIBLE
# ---------------------------------------------------------

selected.extend(
    sample_status(
        by_status,
        "PLAUSIBLE",
        30
    )
)


# ---------------------------------------------------------
# UNSUPPORTED
# ---------------------------------------------------------

selected.extend(
    sample_status(
        by_status,
        "UNSUPPORTED",
        20
    )
)


# ---------------------------------------------------------
# UNVERIFIED
# ---------------------------------------------------------

selected.extend(
    sample_status(
        by_status,
        "UNVERIFIED",
        46
    )
)


# =========================================================
# STEP 6 — VERIFY SAMPLE SIZE
# =========================================================

if len(selected) != 100:

    raise RuntimeError(
        f"Expected exactly 100 examples, "
        f"but selected {len(selected)}"
    )


# =========================================================
# STEP 7 — SHUFFLE
# =========================================================

random.shuffle(selected)


# =========================================================
# STEP 8 — CREATE GOLD-SET RECORDS
# =========================================================

output = []

for i, record in enumerate(
    selected,
    start=201
):

    source = record.get(
        "source",
        record.get(
            "source_mention",
            ""
        )
    )

    relation = record.get(
        "relation",
        record.get(
            "predicted_relation",
            ""
        )
    )

    target = record.get(
        "target",
        record.get(
            "target_mention",
            ""
        )
    )

    sentence = record.get(
        "evidence_sentence",
        record.get(
            "sentence",
            ""
        )
    )

    margin = record.get(
        "margin",
        record.get(
            "svm_margin",
            None
        )
    )

    item = {

        # -------------------------------------------------
        # Gold-set identifier
        # -------------------------------------------------

        "gold_id": i,

        # -------------------------------------------------
        # Relation information
        # -------------------------------------------------

        "source": source,

        "source_type": record.get(
            "source_type",
            ""
        ),

        "relation": relation,

        "target": target,

        "target_type": record.get(
            "target_type",
            ""
        ),

        # -------------------------------------------------
        # Original model information
        # -------------------------------------------------

        "svm_margin": margin,

        "v6_status": record.get(
            "status",
            ""
        ),

        "predicate_status": record.get(
            "predicate_status",
            ""
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

        "sentence": sentence,

        # -------------------------------------------------
        # HUMAN ANNOTATION FIELDS
        #
        # These remain blank initially.
        # We will fill them manually.
        # -------------------------------------------------

        "human_label": "",

        "error_category": "",

        "annotator_notes": ""
    }

    output.append(item)


# =========================================================
# STEP 9 — SAVE
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
            )
            + "\n"
        )


# =========================================================
# STEP 10 — PRINT FINAL STATISTICS
# =========================================================

print()
print("=" * 60)
print("SEMANTIC GOLD SET")
print("=" * 60)

print(
    f"Additional examples : {len(output)}"
)

print(
    f"Gold IDs             : 201–300"
)

print(
    f"Output               : {OUTPUT_FILE}"
)


# ---------------------------------------------------------
# Status distribution
# ---------------------------------------------------------

status_counts = defaultdict(int)

for item in output:

    status_counts[
        item["v6_status"]
    ] += 1


print()
print("Selected V6 status distribution:")

for status in [
    "SUPPORTED",
    "PLAUSIBLE",
    "UNSUPPORTED",
    "UNVERIFIED"
]:

    print(
        f"  {status:12s}: "
        f"{status_counts[status]}"
    )


# ---------------------------------------------------------
# Relation distribution
# ---------------------------------------------------------

relation_counts = defaultdict(int)

for item in output:

    relation_counts[
        item["relation"]
    ] += 1


print()
print("Selected relation distribution:")

for relation, count in sorted(
    relation_counts.items(),
    key=lambda x: (-x[1], x[0])
):

    print(
        f"  {relation:20s}: "
        f"{count}"
    )


# ---------------------------------------------------------
# Margin statistics
# ---------------------------------------------------------

margins = [
    item["svm_margin"]
    for item in output
    if isinstance(
        item["svm_margin"],
        (int, float)
    )
]

if margins:

    print()
    print("SVM margin statistics:")

    print(
        f"  Minimum : {min(margins):.4f}"
    )

    print(
        f"  Maximum : {max(margins):.4f}"
    )

    print(
        f"  Mean    : "
        f"{sum(margins) / len(margins):.4f}"
    )


print()
print("=" * 60)
print("DONE")
print("=" * 60)