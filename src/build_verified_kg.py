import json
import re
from collections import Counter


RAW_GRAPH = "data/semantic_knowledge_graph.json"
OUTPUT_GRAPH = "data/semantic_knowledge_graph_verified.json"


# ============================================================
# CONFIGURATION
# ============================================================

MIN_EVIDENCE_WORDS = 6


# ============================================================
# RELATION-TYPE COMPATIBILITY
# ============================================================
#
# These are quality-control rules, NOT ground truth.
# They are intentionally conservative.
#
# If a relation is not listed here, we do not reject it
# purely on type compatibility.
#

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

    # Convert markdown links to visible text
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Normalize whitespace
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

    normalized = normalize_text(
        text
    )

    if not normalized:
        return 0

    return len(
        normalized.split()
    )


def is_table_or_header(evidence):

    if not evidence:
        return True

    lower = evidence.lower()

    obvious_markers = [

        "| md5 |",

        "| malware",

        "| **c&c**",

        "| **c & c**",

        "malware name",

        "malicious urls",

        "command string",

    ]

    for marker in obvious_markers:

        if marker in lower:
            return True

    # Count markdown table separators
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

    # No rule = don't reject
    if rule is None:
        return True

    return (
        source_type in rule["source"]
        and
        target_type in rule["target"]
    )


# ============================================================
# LOAD GRAPH
# ============================================================

print("=" * 70)
print("BUILDING VERIFIED SEMANTIC CTI KNOWLEDGE GRAPH")
print("=" * 70)

print("\nLoading raw graph...")

with open(
    RAW_GRAPH,
    "r",
    encoding="utf-8"
) as f:

    graph = json.load(f)


print(
    f"Nodes loaded : "
    f"{len(graph['nodes'])}"
)

print(
    f"Edges loaded : "
    f"{len(graph['edges'])}"
)


# ============================================================
# PROCESS EDGES
# ============================================================

verified_edges = []

semantic_total = 0
verified_total = 0
rejected_total = 0

rejection_reasons = Counter()
verified_relations = Counter()
rejected_relations = Counter()


for edge in graph["edges"]:

    relation = edge.get(
        "relation"
    )

    # --------------------------------------------------------
    # Preserve original KG structural edges
    # --------------------------------------------------------

    if relation in {
        "MENTIONED_IN",
        "CANDIDATE_FOR"
    }:

        verified_edges.append(
            edge
        )

        continue


    semantic_total += 1

    properties = edge.get(
        "properties",
        {}
    )

    source = properties.get(
        "source_mention",
        ""
    )

    target = properties.get(
        "target_mention",
        ""
    )

    source_type = properties.get(
        "source_type",
        "UNKNOWN"
    )

    target_type = properties.get(
        "target_type",
        "UNKNOWN"
    )

    evidence = properties.get(
        "evidence_text",
        ""
    )


    # --------------------------------------------------------
    # COLLECT REJECTION REASONS
    # --------------------------------------------------------

    reasons = []


    # --------------------------------------------------------
    # RULE 1: SOURCE MUST APPEAR IN EVIDENCE
    # --------------------------------------------------------

    if not mention_present(
        source,
        evidence
    ):

        reasons.append(
            "SOURCE_NOT_IN_EVIDENCE"
        )


    # --------------------------------------------------------
    # RULE 2: TARGET MUST APPEAR IN EVIDENCE
    # --------------------------------------------------------

    if not mention_present(
        target,
        evidence
    ):

        reasons.append(
            "TARGET_NOT_IN_EVIDENCE"
        )


    # --------------------------------------------------------
    # RULE 3: EVIDENCE LENGTH
    # --------------------------------------------------------

    evidence_words = word_count(
        evidence
    )

    if evidence_words < MIN_EVIDENCE_WORDS:

        reasons.append(
            "VERY_SHORT_EVIDENCE"
        )


    # --------------------------------------------------------
    # RULE 4: TABLE / HEADER
    # --------------------------------------------------------

    if is_table_or_header(
        evidence
    ):

        reasons.append(
            "TABLE_OR_HEADER"
        )


    # --------------------------------------------------------
    # RULE 5: RELATION TYPE COMPATIBILITY
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
    # DECISION
    # --------------------------------------------------------

    if reasons:

        rejected_total += 1

        rejected_relations[
            relation
        ] += 1

        for reason in reasons:

            rejection_reasons[
                reason
            ] += 1

        continue


    # --------------------------------------------------------
    # VERIFIED EDGE
    # --------------------------------------------------------

    verified_edge = dict(
        edge
    )

    verified_properties = dict(
        properties
    )

    verified_properties[
        "validation_status"
    ] = "verified"

    verified_properties[
        "validation_rules"
    ] = [
        "source_present",
        "target_present",
        "adequate_evidence_length",
        "not_table_or_header",
        "compatible_entity_types"
    ]

    verified_edge[
        "properties"
    ] = verified_properties

    verified_edges.append(
        verified_edge
    )

    verified_total += 1

    verified_relations[
        relation
    ] += 1


# ============================================================
# BUILD VERIFIED GRAPH
# ============================================================

verified_graph = {

    "nodes": graph["nodes"],

    "edges": verified_edges,

    "metadata": {

        "source_graph":
            RAW_GRAPH,

        "semantic_edges_before":
            semantic_total,

        "semantic_edges_after":
            verified_total,

        "semantic_edges_rejected":
            rejected_total,

        "margin_threshold":
            0.70,

        "min_evidence_words":
            MIN_EVIDENCE_WORDS,

        "relation_type_filter":
            "conservative",

        "note":
            "Automatically extracted relations "
            "filtered using evidence-presence, "
            "evidence-quality, and entity-type "
            "compatibility rules. This is a "
            "quality-controlled KG, not manually "
            "validated ground truth."
    }
}


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_GRAPH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        verified_graph,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION RESULTS")
print("=" * 70)

print(
    f"Semantic edges before : "
    f"{semantic_total}"
)

print(
    f"Verified edges        : "
    f"{verified_total}"
)

print(
    f"Rejected edges        : "
    f"{rejected_total}"
)


if semantic_total > 0:

    retention = (
        100
        * verified_total
        / semantic_total
    )

    rejection_rate = (
        100
        * rejected_total
        / semantic_total
    )

    print(
        f"Retention             : "
        f"{retention:.2f}%"
    )

    print(
        f"Rejection rate        : "
        f"{rejection_rate:.2f}%"
    )


# ============================================================
# REJECTION REASONS
# ============================================================

print("\n" + "=" * 70)
print("REJECTION REASONS")
print("=" * 70)

for reason, count in (
    rejection_reasons.most_common()
):

    print(
        f"{reason:<35}"
        f"{count}"
    )


# ============================================================
# RELATION COUNTS
# ============================================================

print("\n" + "=" * 70)
print("VERIFIED RELATIONS")
print("=" * 70)

for relation, count in (
    verified_relations.most_common()
):

    print(
        f"{relation:<25}"
        f"{count}"
    )


print("\n" + "=" * 70)
print("REJECTED RELATIONS")
print("=" * 70)

for relation, count in (
    rejected_relations.most_common()
):

    print(
        f"{relation:<25}"
        f"{count}"
    )


# ============================================================
# FINAL GRAPH SIZE
# ============================================================

print("\n" + "=" * 70)
print("VERIFIED GRAPH")
print("=" * 70)

print(
    f"Nodes : "
    f"{len(verified_graph['nodes'])}"
)

print(
    f"Edges : "
    f"{len(verified_graph['edges'])}"
)


print("\nSaved to:")

print(
    OUTPUT_GRAPH
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)