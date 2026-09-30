import json
import random
from collections import defaultdict, Counter

INPUT_FILE = "data/annoctr_relation_predictions_validated_v2.jsonl"
OUTPUT_FILE = "data/validator_v2_audit_sample.json"

RANDOM_SEED = 42

# Number of examples to inspect per category
SUPPORTED_LIMIT = 40
PLAUSIBLE_LIMIT = 50
UNSUPPORTED_LIMIT = 30
UNVERIFIED_LIMIT = 30


def load_records(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def get_audit_fields(record):
    validation = record.get(
        "relation_validation_v2",
        {}
    )

    args = record.get(
        "relation_arguments_v2",
        {}
    )

    return {
        "document": record.get("document"),
        "segment": record.get("segment"),

        "source": record.get("source_mention"),
        "source_type": record.get("source_type"),

        "relation": record.get("predicted_relation"),

        "target": record.get("target_mention"),
        "target_type": record.get("target_type"),

        "margin": record.get("margin"),
        "decision_score": record.get("decision_score"),

        "validation_score":
            validation.get("validation_score"),

        "semantic_score":
            validation.get("semantic_score"),

        "evidence_score":
            validation.get("evidence_score"),

        "argument_score":
            validation.get("argument_score"),

        "status":
            validation.get("status"),

        "argument_status":
            validation.get("argument_status"),

        "type_compatible":
            validation.get("type_compatible"),

        "source_in_evidence":
            validation.get("source_in_evidence"),

        "target_in_evidence":
            validation.get("target_in_evidence"),

        "trigger_in_evidence":
            validation.get("trigger_in_evidence"),

        "trigger":
            args.get("trigger"),

        "direction":
            args.get("direction"),

        "sentence":
            args.get("sentence"),

        "structure":
            args.get("structure"),

        "original_text":
            record.get("text"),

        "reasons":
            validation.get("reasons", [])
    }


def sample_records(records, status, limit):
    candidates = [
        r for r in records
        if r.get(
            "relation_validation_v2",
            {}
        ).get("status") == status
    ]

    # For supported / unsupported, take everything if possible.
    if len(candidates) <= limit:
        return candidates

    random.shuffle(candidates)

    return candidates[:limit]


def print_example(index, item):
    print()
    print("-" * 90)
    print(f"EXAMPLE {index}")
    print("-" * 90)

    print(
        f"{item['source']} "
        f"[{item['source_type']}] "
        f"--{item['relation']}--> "
        f"{item['target']} "
        f"[{item['target_type']}]"
    )

    print(
        f"Status           : {item['status']}"
    )

    print(
        f"Validation score : {item['validation_score']}"
    )

    print(
        f"Semantic score   : {item['semantic_score']}"
    )

    print(
        f"SVM margin       : {item['margin']}"
    )

    print(
        f"Argument status  : {item['argument_status']}"
    )

    print(
        f"Direction        : {item['direction']}"
    )

    print(
        f"Trigger          : {item['trigger']}"
    )

    print(
        f"Structure        : {item['structure']}"
    )

    print(
        f"Type compatible  : {item['type_compatible']}"
    )

    print(
        f"Source in evidence: "
        f"{item['source_in_evidence']}"
    )

    print(
        f"Target in evidence: "
        f"{item['target_in_evidence']}"
    )

    print(
        f"Trigger in evidence: "
        f"{item['trigger_in_evidence']}"
    )

    print()

    print("Selected sentence:")
    print(
        item["sentence"]
        if item["sentence"]
        else "[NONE]"
    )

    print()

    print("Original evidence:")
    print(
        item["original_text"]
        if item["original_text"]
        else "[NONE]"
    )

    print()

    print("Reasons:")

    for reason in item["reasons"]:
        print(
            f"  - {reason}"
        )


def relation_distribution(records):
    counter = Counter()

    for record in records:

        validation = record.get(
            "relation_validation_v2",
            {}
        )

        status = validation.get(
            "status"
        )

        relation = record.get(
            "predicted_relation"
        )

        counter[
            (relation, status)
        ] += 1

    return counter


def main():

    random.seed(
        RANDOM_SEED
    )

    print("=" * 90)
    print("VALIDATOR V2 AUDIT")
    print("=" * 90)

    records = load_records(
        INPUT_FILE
    )

    print(
        f"Records loaded: {len(records)}"
    )

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    status_counts = Counter()

    for record in records:

        status = record.get(
            "relation_validation_v2",
            {}
        ).get("status")

        status_counts[status] += 1

    print()
    print("STATUS DISTRIBUTION")
    print("-" * 90)

    for status, count in (
        status_counts.most_common()
    ):

        percentage = (
            count / len(records)
        ) * 100

        print(
            f"{status:15s} "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Relation × status
    # --------------------------------------------------------

    counter = relation_distribution(
        records
    )

    relations = sorted(
        set(
            r.get(
                "predicted_relation"
            )
            for r in records
        )
    )

    statuses = [
        "SUPPORTED",
        "PLAUSIBLE",
        "UNSUPPORTED",
        "UNVERIFIED"
    ]

    print()
    print("RELATION × STATUS")
    print("-" * 90)

    print(
        f"{'Relation':25s}"
        f"{'Supported':>12s}"
        f"{'Plausible':>12s}"
        f"{'Unsupported':>14s}"
        f"{'Unverified':>12s}"
    )

    for relation in relations:

        print(
            f"{relation:25s}"
            f"{counter[(relation, 'SUPPORTED')]:12d}"
            f"{counter[(relation, 'PLAUSIBLE')]:12d}"
            f"{counter[(relation, 'UNSUPPORTED')]:14d}"
            f"{counter[(relation, 'UNVERIFIED')]:12d}"
        )

    # --------------------------------------------------------
    # Samples
    # --------------------------------------------------------

    sampled = {}

    for status, limit in [
        ("SUPPORTED", SUPPORTED_LIMIT),
        ("PLAUSIBLE", PLAUSIBLE_LIMIT),
        ("UNSUPPORTED", UNSUPPORTED_LIMIT),
        ("UNVERIFIED", UNVERIFIED_LIMIT),
    ]:

        selected = sample_records(
            records,
            status,
            limit
        )

        sampled[status] = [
            get_audit_fields(r)
            for r in selected
        ]

    # --------------------------------------------------------
    # Print supported examples
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("SUPPORTED RELATIONS")
    print("=" * 90)

    for i, item in enumerate(
        sampled["SUPPORTED"],
        1
    ):
        print_example(
            i,
            item
        )

    # --------------------------------------------------------
    # Print plausible examples
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("PLAUSIBLE RELATIONS")
    print("=" * 90)

    for i, item in enumerate(
        sampled["PLAUSIBLE"],
        1
    ):
        print_example(
            i,
            item
        )

    # --------------------------------------------------------
    # Print unsupported examples
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("UNSUPPORTED RELATIONS")
    print("=" * 90)

    for i, item in enumerate(
        sampled["UNSUPPORTED"],
        1
    ):
        print_example(
            i,
            item
        )

    # --------------------------------------------------------
    # Print unverified examples
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("UNVERIFIED RELATIONS")
    print("=" * 90)

    for i, item in enumerate(
        sampled["UNVERIFIED"],
        1
    ):
        print_example(
            i,
            item
        )

    # --------------------------------------------------------
    # Save machine-readable audit
    # --------------------------------------------------------

    audit_output = {
        "input_file": INPUT_FILE,
        "total_records": len(records),

        "status_distribution":
            dict(status_counts),

        "relation_status_distribution": {
            relation: {
                status: counter[
                    (relation, status)
                ]
                for status in statuses
            }
            for relation in relations
        },

        "samples": sampled
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            audit_output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 90)
    print("AUDIT FILE SAVED")
    print("=" * 90)

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()