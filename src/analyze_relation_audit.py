import json
from pathlib import Path
from collections import Counter, defaultdict

INPUT_PATH = Path("data/relation_audit_100_labeled.jsonl")


def pct(x, total):
    return 100 * x / total if total else 0.0


def main():
    records = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if len(records) != 100:
        raise ValueError(f"Expected 100 records, found {len(records)}")

    # ---------------------------------------------------------
    # Overall audit
    # ---------------------------------------------------------

    labels = Counter(r["human_label"] for r in records)
    categories = Counter(r["error_category"] for r in records)

    print("\n" + "=" * 70)
    print("RELATION EXTRACTION AUDIT — 100 EXAMPLES")
    print("=" * 70)

    print("\nOVERALL HUMAN LABELS")
    print("-" * 40)

    for label in ["TRUE", "FALSE", "UNCERTAIN"]:
        n = labels[label]
        print(f"{label:12s}: {n:3d} ({pct(n, len(records)):6.2f}%)")

    # ---------------------------------------------------------
    # V6 status × human label
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("V6 STATUS × HUMAN LABEL")
    print("=" * 70)

    statuses = [
        "SUPPORTED",
        "PLAUSIBLE",
        "UNSUPPORTED",
        "UNVERIFIED",
    ]

    for status in statuses:

        subset = [
            r for r in records
            if r.get("v6_status") == status
        ]

        c = Counter(r["human_label"] for r in subset)

        true_n = c["TRUE"]
        false_n = c["FALSE"]
        uncertain_n = c["UNCERTAIN"]

        print(f"\n{status}")
        print("-" * 40)
        print(f"Total      : {len(subset)}")
        print(
            f"TRUE       : {true_n:3d} "
            f"({pct(true_n, len(subset)):6.2f}%)"
        )
        print(
            f"FALSE      : {false_n:3d} "
            f"({pct(false_n, len(subset)):6.2f}%)"
        )
        print(
            f"UNCERTAIN  : {uncertain_n:3d} "
            f"({pct(uncertain_n, len(subset)):6.2f}%)"
        )

        # Precision among determinate examples
        determinate = true_n + false_n

        if determinate:
            print(
                f"Precision* : "
                f"{pct(true_n, determinate):6.2f}%"
            )

    print("\n*Precision excludes UNCERTAIN cases.")

    # ---------------------------------------------------------
    # Relation × human label
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RELATION × HUMAN LABEL")
    print("=" * 70)

    relation_groups = defaultdict(Counter)

    for r in records:
        relation_groups[r["relation"]][r["human_label"]] += 1

    for relation in sorted(relation_groups):

        c = relation_groups[relation]
        total = sum(c.values())

        print(
            f"{relation:20s} "
            f"n={total:2d}  "
            f"TRUE={c['TRUE']:2d}  "
            f"FALSE={c['FALSE']:2d}  "
            f"UNCERTAIN={c['UNCERTAIN']:2d}"
        )

    # ---------------------------------------------------------
    # Error categories
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ERROR CATEGORIES")
    print("=" * 70)

    for category, n in categories.most_common():

        print(
            f"{category:28s} "
            f"{n:3d} "
            f"({pct(n, len(records)):6.2f}%)"
        )

    # ---------------------------------------------------------
    # Error categories excluding valid/uncertain
    # ---------------------------------------------------------

    actual_errors = [
        r for r in records
        if r["human_label"] == "FALSE"
    ]

    error_categories = Counter(
        r["error_category"]
        for r in actual_errors
    )

    print("\n" + "=" * 70)
    print("ERROR TYPES AMONG FALSE PREDICTIONS")
    print("=" * 70)

    for category, n in error_categories.most_common():

        print(
            f"{category:28s} "
            f"{n:3d} "
            f"({pct(n, len(actual_errors)):6.2f}%)"
        )

    # ---------------------------------------------------------
    # SVM margin analysis
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SVM MARGIN × HUMAN LABEL")
    print("=" * 70)

    bins = [
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.0),
        (1.0, float("inf")),
    ]

    for low, high in bins:

        subset = [
            r for r in records
            if low <= float(r["svm_margin"]) < high
        ]

        c = Counter(r["human_label"] for r in subset)

        print(
            f"{low:.1f}–"
            f"{'∞' if high == float('inf') else f'{high:.1f}':4s} "
            f"n={len(subset):2d}  "
            f"TRUE={c['TRUE']:2d}  "
            f"FALSE={c['FALSE']:2d}  "
            f"UNCERTAIN={c['UNCERTAIN']:2d}"
        )

    # ---------------------------------------------------------
    # V6 decision quality
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("V6 DECISION QUALITY")
    print("=" * 70)

    # Supported = predicted valid
    # Unsupported = predicted invalid
    # Plausible / Unverified = abstention-like states

    supported = [
        r for r in records
        if r["v6_status"] == "SUPPORTED"
    ]

    supported_true = sum(
        r["human_label"] == "TRUE"
        for r in supported
    )

    if supported:
        print(
            f"SUPPORTED precision: "
            f"{pct(supported_true, len(supported)):.2f}%"
        )

    unsupported = [
        r for r in records
        if r["v6_status"] == "UNSUPPORTED"
    ]

    unsupported_false = sum(
        r["human_label"] == "FALSE"
        for r in unsupported
    )

    if unsupported:
        print(
            f"UNSUPPORTED correctness: "
            f"{pct(unsupported_false, len(unsupported)):.2f}%"
        )

    # ---------------------------------------------------------
    # False positives by V6 status
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FALSE PREDICTIONS BY V6 STATUS")
    print("=" * 70)

    false_by_status = Counter(
        r["v6_status"]
        for r in records
        if r["human_label"] == "FALSE"
    )

    for status in statuses:
        print(
            f"{status:15s}: "
            f"{false_by_status[status]:3d}"
        )

    # ---------------------------------------------------------
    # Valid relations
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALID RELATIONS FOUND")
    print("=" * 70)

    valid = [
        r for r in records
        if r["human_label"] == "TRUE"
    ]

    for r in valid:

        print(
            f"\nID {r['audit_id']}: "
            f"{r['source']} "
            f"--{r['relation']}--> "
            f"{r['target']}"
        )

        print(
            f"V6={r['v6_status']} "
            f"margin={float(r['svm_margin']):.3f}"
        )

        print(
            f"Evidence: {r.get('sentence')}"
        )

    # ---------------------------------------------------------
    # Missed valid relations
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALID RELATIONS V6 DID NOT MARK SUPPORTED")
    print("=" * 70)

    missed = [
        r for r in valid
        if r["v6_status"] != "SUPPORTED"
    ]

    print(f"Count: {len(missed)}")

    for r in missed:

        print(
            f"ID {r['audit_id']}: "
            f"{r['source']} "
            f"--{r['relation']}--> "
            f"{r['target']} "
            f"[V6={r['v6_status']}]"
        )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()