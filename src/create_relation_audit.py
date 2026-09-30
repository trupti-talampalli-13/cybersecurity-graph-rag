import json
import random
from pathlib import Path
from collections import defaultdict, Counter


INPUT_PATH = Path(
    "data/annoctr_relation_predictions_validated_v6.jsonl"
)

OUTPUT_PATH = Path(
    "data/relation_audit_100.jsonl"
)

SEED = 42


def load_records():

    records = []

    with INPUT_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def stratified_sample(records):

    random.seed(SEED)

    # --------------------------------------------------------
    # Group by validation status
    # --------------------------------------------------------

    groups = defaultdict(list)

    for record in records:

        groups[
            record.get(
                "status",
                "UNKNOWN"
            )
        ].append(record)

    selected = []

    # --------------------------------------------------------
    # SUPPORTED
    #
    # Take ALL supported examples.
    # There are currently only a few.
    # --------------------------------------------------------

    supported = groups["SUPPORTED"]

    random.shuffle(supported)

    selected.extend(
        supported[:25]
    )

    # --------------------------------------------------------
    # PLAUSIBLE
    # --------------------------------------------------------

    plausible = groups["PLAUSIBLE"]

    random.shuffle(plausible)

    selected.extend(
        plausible[:30]
    )

    # --------------------------------------------------------
    # UNSUPPORTED
    # --------------------------------------------------------

    unsupported = groups["UNSUPPORTED"]

    random.shuffle(unsupported)

    selected.extend(
        unsupported[:25]
    )

    # --------------------------------------------------------
    # UNVERIFIED
    # --------------------------------------------------------

    unverified = groups["UNVERIFIED"]

    random.shuffle(unverified)

    selected.extend(
        unverified[:20]
    )

    # --------------------------------------------------------
    # If SUPPORTED has fewer than 25, compensate using
    # additional uncertain examples.
    # --------------------------------------------------------

    target_size = 100

    if len(selected) < target_size:

        already = {
            (
                r.get("document"),
                r.get("segment"),
                r.get("source"),
                r.get("relation"),
                r.get("target")
            )
            for r in selected
        }

        remaining = [
            r
            for r in records
            if (
                r.get("document"),
                r.get("segment"),
                r.get("source"),
                r.get("relation"),
                r.get("target")
            )
            not in already
        ]

        random.shuffle(
            remaining
        )

        selected.extend(
            remaining[
                :target_size - len(selected)
            ]
        )

    return selected[:target_size]


def make_audit_record(
    record,
    audit_id
):

    return {

        "audit_id":
            audit_id,

        "document":
            record.get(
                "document"
            ),

        "segment":
            record.get(
                "segment"
            ),

        "source":
            record.get(
                "source",
                record.get(
                    "source_mention",
                    ""
                )
            ),

        "source_type":
            record.get(
                "source_type",
                ""
            ),

        "relation":
            record.get(
                "relation",
                record.get(
                    "predicted_relation",
                    ""
                )
            ),

        "target":
            record.get(
                "target",
                record.get(
                    "target_mention",
                    ""
                )
            ),

        "target_type":
            record.get(
                "target_type",
                ""
            ),

        "sentence":
            record.get(
                "evidence_sentence",
                record.get(
                    "sentence",
                    ""
                )
            ),

        "v6_status":
            record.get(
                "status",
                ""
            ),

        "svm_margin":
            record.get(
                "svm_margin",
                record.get(
                    "margin",
                    None
                )
            ),

        "predicate":
            record.get(
                "predicate"
            ),

        "predicate_structure":
            record.get(
                "predicate_structure"
            ),

        "predicate_argument_details":
            record.get(
                "predicate_argument_details",
                {}
            ),

        # ====================================================
        # HUMAN LABEL
        # ====================================================
        #
        # Fill this manually:
        #
        # TRUE
        # FALSE
        # UNCERTAIN
        #
        "human_label":
            "",

        # Optional error category.
        #
        # Examples:
        #
        # WRONG_SUBJECT
        # WRONG_OBJECT
        # PASSIVE_REVERSAL
        # COORDINATION_ERROR
        # NESTED_ENTITY
        # REPORTING_CONTEXT
        # COMPARISON
        # TABLE
        # SEMANTIC_MISMATCH
        # OTHER
        #
        "error_category":
            "",

        "review_notes":
            ""
    }


def main():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}"
        )

    records = load_records()

    print(
        f"Loaded {len(records)} records"
    )

    selected = stratified_sample(
        records
    )

    print(
        f"Selected {len(selected)} audit records"
    )

    # --------------------------------------------------------
    # Shuffle final audit set
    # --------------------------------------------------------

    random.seed(SEED)

    random.shuffle(
        selected
    )

    audit_records = []

    for i, record in enumerate(
        selected,
        1
    ):

        audit_records.append(
            make_audit_record(
                record,
                i
            )
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    counts = Counter(
        r["v6_status"]
        for r in audit_records
    )

    print()
    print("=" * 60)
    print("AUDIT SAMPLE")
    print("=" * 60)

    for status in [
        "SUPPORTED",
        "PLAUSIBLE",
        "UNSUPPORTED",
        "UNVERIFIED"
    ]:

        print(
            f"{status:15s}: "
            f"{counts[status]}"
        )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()