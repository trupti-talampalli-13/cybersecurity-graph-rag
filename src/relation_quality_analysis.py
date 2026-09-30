import json
import re
from collections import Counter, defaultdict

GRAPH_FILE = "data/semantic_knowledge_graph.json"


# ============================================================
# LOAD GRAPH
# ============================================================

print("=" * 70)
print("RELATION QUALITY ANALYSIS")
print("=" * 70)

print("\nLoading semantic knowledge graph...")

with open(GRAPH_FILE, "r", encoding="utf-8") as f:
    graph = json.load(f)

print(f"Nodes loaded : {len(graph['nodes'])}")
print(f"Edges loaded : {len(graph['edges'])}")


# ============================================================
# GET SEMANTIC RELATIONS
# ============================================================

relation_edges = [
    edge
    for edge in graph["edges"]
    if edge.get("relation") not in {
        "MENTIONED_IN",
        "CANDIDATE_FOR"
    }
]

print(
    f"Semantic relation edges: "
    f"{len(relation_edges)}"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_text(text):
    """
    Normalize text for simple evidence matching.
    """
    if not text:
        return ""

    text = str(text).lower()

    # Remove markdown links
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

    # Collapse whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def mention_present(mention, evidence):
    """
    Check whether an entity mention appears in
    the evidence text after normalization.
    """

    mention_norm = normalize_text(
        mention
    )

    evidence_norm = normalize_text(
        evidence
    )

    if not mention_norm or not evidence_norm:
        return False

    return mention_norm in evidence_norm


def evidence_word_count(evidence):
    """
    Count words in evidence.
    """

    if not evidence:
        return 0

    return len(
        normalize_text(evidence).split()
    )


def looks_like_table_or_header(evidence):
    """
    Detect obvious table/header fragments.
    This is only a flag, NOT a rejection rule.
    """

    if not evidence:
        return False

    evidence_lower = evidence.lower()

    table_markers = [
        "| md5 |",
        "| malware",
        "| **c&c**",
        "| **c & c**",
        "| c&c |",
        "malware name",
        "command string",
        "###",
        "## "
    ]

    for marker in table_markers:

        if marker in evidence_lower:
            return True

    # Very short evidence is also suspicious
    words = normalize_text(
        evidence
    ).split()

    if len(words) <= 5:
        return True

    return False


# ============================================================
# ANALYSIS COUNTERS
# ============================================================

relation_counts = Counter()

relation_margin_stats = defaultdict(list)

source_missing = Counter()
target_missing = Counter()

both_present = Counter()
neither_present = Counter()

short_evidence = Counter()
table_evidence = Counter()

suspicious_edges = []

same_type_edges = Counter()

relation_type_pairs = Counter()


# ============================================================
# ANALYZE EACH RELATION
# ============================================================

for edge in relation_edges:

    relation = edge.get(
        "relation",
        "UNKNOWN"
    )

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

    margin = properties.get(
        "margin",
        0.0
    )


    # --------------------------------------------------------
    # Relation count
    # --------------------------------------------------------

    relation_counts[
        relation
    ] += 1


    # --------------------------------------------------------
    # Margin
    # --------------------------------------------------------

    relation_margin_stats[
        relation
    ].append(margin)


    # --------------------------------------------------------
    # Type pairs
    # --------------------------------------------------------

    relation_type_pairs[
        (
            relation,
            source_type,
            target_type
        )
    ] += 1


    # --------------------------------------------------------
    # Same entity type
    # --------------------------------------------------------

    if (
        source_type == target_type
    ):

        same_type_edges[
            (
                relation,
                source_type
            )
        ] += 1


    # --------------------------------------------------------
    # Evidence presence
    # --------------------------------------------------------

    source_found = mention_present(
        source,
        evidence
    )

    target_found = mention_present(
        target,
        evidence
    )


    if source_found and target_found:

        both_present[
            relation
        ] += 1

    elif not source_found and not target_found:

        neither_present[
            relation
        ] += 1

    elif not source_found:

        source_missing[
            relation
        ] += 1

    elif not target_found:

        target_missing[
            relation
        ] += 1


    # --------------------------------------------------------
    # Evidence length
    # --------------------------------------------------------

    word_count = evidence_word_count(
        evidence
    )

    if word_count <= 5:

        short_evidence[
            relation
        ] += 1


    # --------------------------------------------------------
    # Table/header detection
    # --------------------------------------------------------

    if looks_like_table_or_header(
        evidence
    ):

        table_evidence[
            relation
        ] += 1


    # --------------------------------------------------------
    # Suspicious edge flag
    # --------------------------------------------------------

    flags = []

    if not source_found:
        flags.append(
            "SOURCE_NOT_IN_EVIDENCE"
        )

    if not target_found:
        flags.append(
            "TARGET_NOT_IN_EVIDENCE"
        )

    if word_count <= 5:
        flags.append(
            "VERY_SHORT_EVIDENCE"
        )

    if looks_like_table_or_header(
        evidence
    ):
        flags.append(
            "TABLE_OR_HEADER"
        )

    if source_type == target_type:
        flags.append(
            "SAME_ENTITY_TYPE"
        )


    if flags:

        suspicious_edges.append({

            "relation": relation,

            "source": source,

            "target": target,

            "source_type":
                source_type,

            "target_type":
                target_type,

            "margin":
                margin,

            "flags":
                flags,

            "document":
                properties.get(
                    "document",
                    ""
                ),

            "segment":
                properties.get(
                    "segment",
                    ""
                ),

            "evidence":
                evidence
        })


# ============================================================
# RELATION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RELATION DISTRIBUTION")
print("=" * 70)

for relation, count in relation_counts.most_common():

    percentage = (
        100 * count / len(relation_edges)
    )

    print(
        f"{relation:<25}"
        f"{count:>5}"
        f" ({percentage:6.2f}%)"
    )


# ============================================================
# EVIDENCE PRESENCE
# ============================================================

print("\n" + "=" * 70)
print("ENTITY PRESENCE IN EVIDENCE")
print("=" * 70)

total = len(relation_edges)

both = sum(
    both_present.values()
)

src_missing = sum(
    source_missing.values()
)

tgt_missing = sum(
    target_missing.values()
)

neither = sum(
    neither_present.values()
)


print(
    f"Both source + target present : "
    f"{both} "
    f"({100 * both / total:.2f}%)"
)

print(
    f"Source missing               : "
    f"{src_missing} "
    f"({100 * src_missing / total:.2f}%)"
)

print(
    f"Target missing               : "
    f"{tgt_missing} "
    f"({100 * tgt_missing / total:.2f}%)"
)

print(
    f"Both missing                 : "
    f"{neither} "
    f"({100 * neither / total:.2f}%)"
)


# ============================================================
# EVIDENCE QUALITY
# ============================================================

print("\n" + "=" * 70)
print("EVIDENCE QUALITY FLAGS")
print("=" * 70)

short_total = sum(
    short_evidence.values()
)

table_total = sum(
    table_evidence.values()
)

print(
    f"Very short evidence (<=5 words): "
    f"{short_total} "
    f"({100 * short_total / total:.2f}%)"
)

print(
    f"Table/header-like evidence      : "
    f"{table_total} "
    f"({100 * table_total / total:.2f}%)"
)


# ============================================================
# MARGIN BY RELATION
# ============================================================

print("\n" + "=" * 70)
print("MARGIN BY RELATION")
print("=" * 70)


for relation, values in (
    sorted(
        relation_margin_stats.items(),
        key=lambda x: -len(x[1])
    )
):

    values = sorted(values)

    median = values[
        len(values) // 2
    ]

    mean = sum(values) / len(values)

    print(
        f"{relation:<25}"
        f"n={len(values):>4} "
        f"mean={mean:.4f} "
        f"median={median:.4f} "
        f"min={min(values):.4f} "
        f"max={max(values):.4f}"
    )


# ============================================================
# ENTITY TYPE PAIRS
# ============================================================

print("\n" + "=" * 70)
print("TOP ENTITY TYPE PAIRS")
print("=" * 70)


for (
    relation,
    source_type,
    target_type
), count in relation_type_pairs.most_common(40):

    print(
        f"{relation:<22}"
        f"{source_type:<18} → "
        f"{target_type:<18}"
        f"{count}"
    )


# ============================================================
# SAME TYPE RELATIONS
# ============================================================

print("\n" + "=" * 70)
print("SAME-TYPE RELATIONS")
print("=" * 70)


for (
    relation,
    entity_type
), count in same_type_edges.most_common():

    print(
        f"{relation:<22}"
        f"{entity_type:<18}"
        f"{count}"
    )


# ============================================================
# SUSPICIOUS EDGE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUSPICIOUS EDGE SUMMARY")
print("=" * 70)


flag_counts = Counter()

for edge in suspicious_edges:

    for flag in edge["flags"]:

        flag_counts[flag] += 1


print(
    f"Edges with at least one flag : "
    f"{len(suspicious_edges)} "
    f"({100 * len(suspicious_edges) / total:.2f}%)"
)


for flag, count in flag_counts.most_common():

    print(
        f"{flag:<30}"
        f"{count}"
    )


# ============================================================
# EXAMPLES OF SUSPICIOUS EDGES
# ============================================================

print("\n" + "=" * 70)
print("SUSPICIOUS EDGE EXAMPLES")
print("=" * 70)


# Show up to 20 examples

for i, edge in enumerate(
    suspicious_edges[:20],
    start=1
):

    print(
        f"\nExample {i}"
    )

    print(
        f"Relation : "
        f"{edge['relation']}"
    )

    print(
        f"Source   : "
        f"{edge['source']} "
        f"[{edge['source_type']}]"
    )

    print(
        f"Target   : "
        f"{edge['target']} "
        f"[{edge['target_type']}]"
    )

    print(
        f"Margin   : "
        f"{edge['margin']:.4f}"
    )

    print(
        f"Flags    : "
        f"{', '.join(edge['flags'])}"
    )

    print(
        f"Document : "
        f"{edge['document']}"
    )

    print(
        f"Segment  : "
        f"{edge['segment']}"
    )

    evidence = edge["evidence"]

    if len(evidence) > 400:

        evidence = (
            evidence[:400]
            + "..."
        )

    print(
        f"Evidence : "
        f"{evidence}"
    )


# ============================================================
# SAVE ANALYSIS
# ============================================================

OUTPUT_FILE = (
    "data/relation_quality_report.json"
)

report = {

    "total_semantic_edges":
        total,

    "relation_distribution":
        dict(relation_counts),

    "evidence_presence": {

        "both_present":
            both,

        "source_missing":
            src_missing,

        "target_missing":
            tgt_missing,

        "both_missing":
            neither
    },

    "evidence_quality": {

        "very_short":
            short_total,

        "table_or_header":
            table_total
    },

    "same_type_relations":
        {
            f"{r}|{t}": c
            for (
                r,
                t
            ), c in same_type_edges.items()
        },

    "suspicious_edges":
        suspicious_edges
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nDetailed report saved to:"
)

print(
    OUTPUT_FILE
)