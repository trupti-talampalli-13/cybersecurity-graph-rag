import json
import re
from collections import Counter, defaultdict


RAW_GRAPH = "data/semantic_knowledge_graph.json"
OUTPUT_FILE = "data/rejected_relation_audit.json"


MIN_EVIDENCE_WORDS = 6


# ============================================================
# SAME COMPATIBILITY RULES USED BY build_verified_kg.py
# ============================================================

RELATION_COMPATIBILITY = {

    "uses": {
        "source": {
            "GROUP",
            "ORG",
            "MALWARE",
            "TOOL",
            "TECHNIQUE",
            "TACTIC"
        },
        "target": {
            "MALWARE",
            "TOOL",
            "TECHNIQUE",
            "TACTIC",
            "CON"
        }
    },

    "targets": {
        "source": {
            "GROUP",
            "ORG",
            "MALWARE",
            "TOOL"
        },
        "target": {
            "ORG",
            "SECTOR",
            "LOC",
            "MALWARE",
            "CON"
        }
    },

    "originates-from": {
        "source": {
            "GROUP",
            "ORG",
            "MALWARE",
            "TOOL"
        },
        "target": {
            "LOC"
        }
    },

    "located-at": {
        "source": {
            "GROUP",
            "ORG",
            "MALWARE",
            "TOOL"
        },
        "target": {
            "LOC"
        }
    },

    "exfiltrates-to": {
        "source": {
            "GROUP",
            "ORG",
            "MALWARE",
            "TOOL"
        },
        "target": {
            "ORG",
            "LOC",
            "CON",
            "TOOL"
        }
    },

    "communicates-with": {
        "source": {
            "GROUP",
            "ORG",
            "MALWARE",
            "TOOL"
        },
        "target": {
            "GROUP",
            "ORG",
            "MALWARE",
            "TOOL"
        }
    },

    "downloads": {
        "source": {
            "MALWARE",
            "TOOL"
        },
        "target": {
            "MALWARE",
            "TOOL"
        }
    },

    "attributed-to": {
        "source": {
            "GROUP",
            "ORG",
            "MALWARE",
            "TOOL",
            "CON",
            "TECHNIQUE"
        },
        "target": {
            "GROUP",
            "ORG"
        }
    }
}


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def mention_present(mention, evidence):

    mention_norm = normalize_text(
        mention
    )

    evidence_norm = normalize_text(
        evidence
    )

    if not mention_norm:
        return False

    return mention_norm in evidence_norm


def word_count(text):

    text = normalize_text(text)

    if not text:
        return 0

    return len(text.split())


def is_table_or_header(evidence):

    if not evidence:
        return True

    lower = evidence.lower()

    markers = [
        "| md5 |",
        "| malware",
        "| **c&c**",
        "| **c & c**",
        "malware name",
        "malicious urls",
        "command string"
    ]

    for marker in markers:

        if marker in lower:
            return True

    if (
        "|" in evidence
        and evidence.count("|") >= 3
    ):
        return True

    return False


def type_compatible(
    relation,
    source_type,
    target_type
):

    rule = RELATION_COMPATIBILITY.get(
        relation
    )

    if rule is None:
        return True

    return (
        source_type in rule["source"]
        and
        target_type in rule["target"]
    )


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("AUDITING REJECTED SEMANTIC RELATIONS")
print("=" * 70)

with open(
    RAW_GRAPH,
    "r",
    encoding="utf-8"
) as f:

    graph = json.load(f)


# ============================================================
# ANALYSIS CONTAINERS
# ============================================================

rejected = []

reason_counts = Counter()

relation_counts = Counter()

relation_reason_counts = Counter()

type_pair_counts = Counter()

relation_type_pair_counts = Counter()

examples = defaultdict(list)


# ============================================================
# INSPECT EVERY SEMANTIC EDGE
# ============================================================

for edge in graph["edges"]:

    relation = edge.get(
        "relation"
    )

    if relation in {
        "MENTIONED_IN",
        "CANDIDATE_FOR"
    }:

        continue


    props = edge.get(
        "properties",
        {}
    )

    source = props.get(
        "source_mention",
        ""
    )

    target = props.get(
        "target_mention",
        ""
    )

    source_type = props.get(
        "source_type",
        "UNKNOWN"
    )

    target_type = props.get(
        "target_type",
        "UNKNOWN"
    )

    evidence = props.get(
        "evidence_text",
        ""
    )

    reasons = []


    # --------------------------------------------------------
    # Evidence checks
    # --------------------------------------------------------

    if not mention_present(
        source,
        evidence
    ):

        reasons.append(
            "SOURCE_NOT_IN_EVIDENCE"
        )


    if not mention_present(
        target,
        evidence
    ):

        reasons.append(
            "TARGET_NOT_IN_EVIDENCE"
        )


    if word_count(
        evidence
    ) < MIN_EVIDENCE_WORDS:

        reasons.append(
            "VERY_SHORT_EVIDENCE"
        )


    if is_table_or_header(
        evidence
    ):

        reasons.append(
            "TABLE_OR_HEADER"
        )


    # --------------------------------------------------------
    # Type compatibility
    # --------------------------------------------------------

    if not type_compatible(
        relation,
        source_type,
        target_type
    ):

        reasons.append(
            "INCOMPATIBLE_ENTITY_TYPES"
        )


    # --------------------------------------------------------
    # Only retain rejected relations
    # --------------------------------------------------------

    if not reasons:
        continue


    record = {

        "edge_id":
            edge.get("id"),

        "relation":
            relation,

        "source":
            source,

        "source_type":
            source_type,

        "target":
            target,

        "target_type":
            target_type,

        "evidence":
            evidence,

        "margin":
            props.get("margin"),

        "decision_score":
            props.get("decision_score"),

        "reasons":
            reasons
    }

    rejected.append(record)


    # ========================================================
    # COUNTERS
    # ========================================================

    relation_counts[
        relation
    ] += 1


    for reason in reasons:

        reason_counts[
            reason
        ] += 1

        relation_reason_counts[
            (
                relation,
                reason
            )
        ] += 1


    pair = (
        source_type,
        target_type
    )

    type_pair_counts[
        pair
    ] += 1


    relation_type_pair_counts[
        (
            relation,
            source_type,
            target_type
        )
    ] += 1


    # Store a few representative examples
    key = (
        relation,
        source_type,
        target_type
    )

    if len(examples[key]) < 5:

        examples[key].append(
            record
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Rejected semantic edges : "
    f"{len(rejected)}"
)


# ============================================================
# REJECTION REASONS
# ============================================================

print("\n" + "=" * 70)
print("REJECTION REASONS")
print("=" * 70)

for reason, count in (
    reason_counts.most_common()
):

    print(
        f"{reason:<35}"
        f"{count}"
    )


# ============================================================
# REJECTIONS BY RELATION
# ============================================================

print("\n" + "=" * 70)
print("REJECTIONS BY RELATION")
print("=" * 70)

for relation, count in (
    relation_counts.most_common()
):

    print(
        f"{relation:<25}"
        f"{count}"
    )


# ============================================================
# RELATION × REASON
# ============================================================

print("\n" + "=" * 70)
print("RELATION × REJECTION REASON")
print("=" * 70)

for (
    relation,
    reason
), count in sorted(
    relation_reason_counts.items(),
    key=lambda x: -x[1]
):

    print(
        f"{relation:<22}"
        f"{reason:<35}"
        f"{count}"
    )


# ============================================================
# TOP TYPE PAIRS
# ============================================================

print("\n" + "=" * 70)
print("TOP REJECTED ENTITY TYPE PAIRS")
print("=" * 70)

for (
    source_type,
    target_type
), count in type_pair_counts.most_common(40):

    print(
        f"{source_type:<15}"
        f"→ "
        f"{target_type:<15}"
        f"{count}"
    )


# ============================================================
# RELATION × TYPE PAIRS
# ============================================================

print("\n" + "=" * 70)
print("TOP RELATION × ENTITY TYPE PAIRS")
print("=" * 70)

for (
    relation,
    source_type,
    target_type
), count in sorted(
    relation_type_pair_counts.items(),
    key=lambda x: -x[1]
)[:50]:

    print(
        f"{relation:<22}"
        f"{source_type:<15}"
        f"→ "
        f"{target_type:<15}"
        f"{count}"
    )


# ============================================================
# REPRESENTATIVE EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("REPRESENTATIVE REJECTED EXAMPLES")
print("=" * 70)

for key in sorted(
    examples.keys()
):

    relation, source_type, target_type = key

    print(
        "\n"
        + "-" * 70
    )

    print(
        f"Relation: {relation}"
    )

    print(
        f"Types: "
        f"{source_type} → {target_type}"
    )

    for i, example in enumerate(
        examples[key],
        start=1
    ):

        print(
            f"\nExample {i}:"
        )

        print(
            f"  Source   : "
            f"{example['source']}"
        )

        print(
            f"  Target   : "
            f"{example['target']}"
        )

        print(
            f"  Margin   : "
            f"{example['margin']}"
        )

        print(
            f"  Reasons  : "
            f"{', '.join(example['reasons'])}"
        )

        print(
            f"  Evidence : "
            f"{example['evidence']}"
        )


# ============================================================
# SAVE FULL AUDIT
# ============================================================

audit = {

    "summary": {

        "rejected_edges":
            len(rejected),

        "rejection_reasons":
            dict(reason_counts),

        "relations":
            dict(relation_counts)
    },

    "relation_reason_counts": {

        f"{relation} | {reason}":
            count

        for (
            relation,
            reason
        ), count in relation_reason_counts.items()
    },

    "type_pair_counts": {

        f"{source} -> {target}":
            count

        for (
            source,
            target
        ), count in type_pair_counts.items()
    },

    "relation_type_pair_counts": {

        f"{relation} | {source} -> {target}":
            count

        for (
            relation,
            source,
            target
        ), count in relation_type_pair_counts.items()
    },

    "representative_examples": {

        f"{relation} | {source_type} -> {target_type}":
            records

        for (
            relation,
            source_type,
            target_type
        ), records in examples.items()
    },

    "all_rejected_edges":
        rejected
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        audit,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n" + "=" * 70)
print("AUDIT SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)

print("=" * 70)