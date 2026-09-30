import json
from pathlib import Path
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/annoctr_relation_predictions_linguistic.jsonl"
)

TOP_N = 20

# We focus on relations that actually have predictions.
RELATIONS = [
    "uses",
    "targets",
    "attributed-to",
    "originates-from",
    "communicates-with",
    "exfiltrates-to",
    "located-at",
    "downloads",
    "drops",
    "impersonates",
    "variant-of",
    "exploits",
    "delivers",
]


# ============================================================
# LOAD
# ============================================================

def load_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                records.append(json.loads(line))

    return records


# ============================================================
# PRINT RECORD
# ============================================================

def print_record(record, rank=None):

    analysis = record.get(
        "linguistic_analysis",
        {}
    )

    relation = record.get(
        "predicted_relation",
        "UNKNOWN"
    )

    source = record.get(
        "source_mention",
        ""
    )

    target = record.get(
        "target_mention",
        ""
    )

    text = record.get(
        "text",
        ""
    )

    margin = record.get(
        "margin",
        None
    )

    decision_score = record.get(
        "decision_score",
        None
    )

    linguistic_score = analysis.get(
        "linguistic_score",
        0.0
    )

    triggers = analysis.get(
        "relation_triggers",
        []
    )

    source_dep = analysis.get(
        "source_dep"
    )

    target_dep = analysis.get(
        "target_dep"
    )

    source_pos = analysis.get(
        "source_pos"
    )

    target_pos = analysis.get(
        "target_pos"
    )

    source_subject = analysis.get(
        "source_is_subject",
        False
    )

    target_object = analysis.get(
        "target_is_object",
        False
    )

    path_verb = analysis.get(
        "path_has_verb",
        False
    )

    path_prep = analysis.get(
        "path_has_preposition",
        False
    )

    same_sentence = analysis.get(
        "same_sentence",
        False
    )

    token_distance = analysis.get(
        "token_distance"
    )

    dependency_distance = analysis.get(
        "dependency_distance"
    )

    dependency_path = analysis.get(
        "dependency_path_text",
        ""
    )

    if rank is not None:
        print(
            f"\n{'-' * 90}"
        )
        print(
            f"RANK {rank}"
        )
        print(
            f"{'-' * 90}"
        )

    print(
        f"RELATION      : {source} "
        f"--[{relation}]--> {target}"
    )

    print(
        f"SVM margin    : {margin}"
    )

    print(
        f"SVM score     : {decision_score}"
    )

    print(
        f"Linguistic    : {linguistic_score}"
    )

    print(
        f"Trigger       : {triggers}"
    )

    print(
        f"Source dep    : {source_dep} "
        f"({source_pos})"
    )

    print(
        f"Target dep    : {target_dep} "
        f"({target_pos})"
    )

    print(
        f"Source subject: {source_subject}"
    )

    print(
        f"Target object  : {target_object}"
    )

    print(
        f"Path has verb  : {path_verb}"
    )

    print(
        f"Path has prep  : {path_prep}"
    )

    print(
        f"Same sentence : {same_sentence}"
    )

    print(
        f"Token distance: {token_distance}"
    )

    print(
        f"Dependency dist: {dependency_distance}"
    )

    print(
        f"Dependency path:\n  {dependency_path}"
    )

    print(
        f"\nTEXT:\n{text}"
    )


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def average(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


def summarize_relation(records):

    margins = []
    linguistic_scores = []

    trigger_count = 0
    subject_count = 0
    object_count = 0
    verb_count = 0
    same_sentence_count = 0

    for record in records:

        analysis = record.get(
            "linguistic_analysis",
            {}
        )

        if record.get("margin") is not None:
            margins.append(
                record["margin"]
            )

        linguistic_scores.append(
            analysis.get(
                "linguistic_score",
                0.0
            )
        )

        if analysis.get(
            "relation_triggers"
        ):
            trigger_count += 1

        if analysis.get(
            "source_is_subject"
        ):
            subject_count += 1

        if analysis.get(
            "target_is_object"
        ):
            object_count += 1

        if analysis.get(
            "path_has_verb"
        ):
            verb_count += 1

        if analysis.get(
            "same_sentence"
        ):
            same_sentence_count += 1

    n = len(records)

    return {
        "n": n,
        "avg_margin": average(margins),
        "avg_linguistic_score": average(
            linguistic_scores
        ),
        "trigger_pct": (
            100 * trigger_count / n
            if n else 0
        ),
        "source_subject_pct": (
            100 * subject_count / n
            if n else 0
        ),
        "target_object_pct": (
            100 * object_count / n
            if n else 0
        ),
        "verb_path_pct": (
            100 * verb_count / n
            if n else 0
        ),
        "same_sentence_pct": (
            100 * same_sentence_count / n
            if n else 0
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("LINGUISTIC RELATION AUDIT")
    print("=" * 90)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}"
        )

    records = load_jsonl(
        INPUT_FILE
    )

    print(
        f"Records loaded: {len(records)}"
    )

    # --------------------------------------------------------
    # Group by relation
    # --------------------------------------------------------

    grouped = defaultdict(list)

    for record in records:

        relation = record.get(
            "predicted_relation"
        )

        grouped[relation].append(
            record
        )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("RELATION SUMMARY")
    print("=" * 90)

    print(
        f"{'Relation':20s}"
        f"{'N':>7s}"
        f"{'Margin':>10s}"
        f"{'Ling':>10s}"
        f"{'Trigger':>10s}"
        f"{'Subj':>10s}"
        f"{'Obj':>10s}"
        f"{'VerbPath':>10s}"
    )

    print("-" * 90)

    for relation in RELATIONS:

        relation_records = grouped.get(
            relation,
            []
        )

        if not relation_records:
            continue

        stats = summarize_relation(
            relation_records
        )

        print(
            f"{relation:20s}"
            f"{stats['n']:7d}"
            f"{stats['avg_margin']:10.3f}"
            f"{stats['avg_linguistic_score']:10.3f}"
            f"{stats['trigger_pct']:9.1f}%"
            f"{stats['source_subject_pct']:9.1f}%"
            f"{stats['target_object_pct']:9.1f}%"
            f"{stats['verb_path_pct']:9.1f}%"
        )

    # --------------------------------------------------------
    # Inspect strongest / weakest examples
    # --------------------------------------------------------

    for relation in RELATIONS:

        relation_records = grouped.get(
            relation,
            []
        )

        if not relation_records:
            continue

        print()
        print()
        print("#" * 90)
        print(
            f"RELATION: {relation.upper()}"
        )
        print("#" * 90)

        # ----------------------------------------------------
        # Highest linguistic support
        # ----------------------------------------------------

        strongest = sorted(
            relation_records,
            key=lambda x: x.get(
                "linguistic_analysis",
                {}
            ).get(
                "linguistic_score",
                0.0
            ),
            reverse=True
        )[:TOP_N]

        print()
        print(
            "=" * 90
        )
        print(
            f"TOP {TOP_N} STRONGEST LINGUISTIC EXAMPLES"
        )
        print(
            "=" * 90
        )

        for rank, record in enumerate(
            strongest,
            start=1
        ):

            print_record(
                record,
                rank
            )

        # ----------------------------------------------------
        # Weakest linguistic support
        # ----------------------------------------------------

        weakest = sorted(
            relation_records,
            key=lambda x: x.get(
                "linguistic_analysis",
                {}
            ).get(
                "linguistic_score",
                0.0
            )
        )[:TOP_N]

        print()
        print(
            "=" * 90
        )
        print(
            f"TOP {TOP_N} WEAKEST LINGUISTIC EXAMPLES"
        )
        print(
            "=" * 90
        )

        for rank, record in enumerate(
            weakest,
            start=1
        ):

            print_record(
                record,
                rank
            )

    # --------------------------------------------------------
    # Particularly important:
    # High SVM confidence + low linguistic support
    # --------------------------------------------------------

    print()
    print()
    print("#" * 90)
    print(
        "HIGH SVM CONFIDENCE + LOW LINGUISTIC SUPPORT"
    )
    print("#" * 90)

    suspicious = []

    for record in records:

        analysis = record.get(
            "linguistic_analysis",
            {}
        )

        margin = record.get(
            "margin",
            0.0
        )

        linguistic_score = analysis.get(
            "linguistic_score",
            0.0
        )

        # Strong SVM confidence but weak
        # linguistic evidence.
        if (
            margin >= 0.70
            and linguistic_score < 0.40
        ):
            suspicious.append(record)

    suspicious.sort(
        key=lambda x: (
            x.get("margin", 0.0),
            -x.get(
                "linguistic_analysis",
                {}
            ).get(
                "linguistic_score",
                0.0
            )
        ),
        reverse=True
    )

    print(
        f"Found {len(suspicious)} "
        f"high-margin / low-linguistic cases."
    )

    for rank, record in enumerate(
        suspicious[:50],
        start=1
    ):

        print_record(
            record,
            rank
        )

    # --------------------------------------------------------
    # Low SVM confidence + high linguistic support
    # --------------------------------------------------------

    print()
    print()
    print("#" * 90)
    print(
        "LOW SVM CONFIDENCE + HIGH LINGUISTIC SUPPORT"
    )
    print("#" * 90)

    interesting = []

    for record in records:

        analysis = record.get(
            "linguistic_analysis",
            {}
        )

        margin = record.get(
            "margin",
            0.0
        )

        linguistic_score = analysis.get(
            "linguistic_score",
            0.0
        )

        if (
            margin < 0.70
            and linguistic_score >= 0.60
        ):
            interesting.append(record)

    interesting.sort(
        key=lambda x: (
            x.get(
                "linguistic_analysis",
                {}
            ).get(
                "linguistic_score",
                0.0
            ),
            -x.get("margin", 0.0)
        ),
        reverse=True
    )

    print(
        f"Found {len(interesting)} "
        f"low-margin / high-linguistic cases."
    )

    for rank, record in enumerate(
        interesting[:50],
        start=1
    ):

        print_record(
            record,
            rank
        )

    print()
    print("=" * 90)
    print("AUDIT COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()