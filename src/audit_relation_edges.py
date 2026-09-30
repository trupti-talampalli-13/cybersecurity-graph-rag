import json
from collections import Counter, defaultdict

GRAPH_FILE = "data/semantic_knowledge_graph.json"


# ============================================================
# LOAD GRAPH
# ============================================================

with open(GRAPH_FILE, "r", encoding="utf-8") as f:
    graph = json.load(f)

nodes = {
    str(node["id"]): node
    for node in graph["nodes"]
}


# ============================================================
# GET SEMANTIC RELATION EDGES
# ============================================================

relation_edges = [
    edge
    for edge in graph["edges"]
    if edge.get("relation") not in {
        "MENTIONED_IN",
        "CANDIDATE_FOR"
    }
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SEMANTIC KNOWLEDGE GRAPH AUDIT")
print("=" * 70)

print(
    f"\nSemantic relation edges: {len(relation_edges)}"
)


# ============================================================
# RELATION DISTRIBUTION
# ============================================================

relation_counts = Counter(
    edge["relation"]
    for edge in relation_edges
)

print("\nRelation distribution:")

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
# NODE RESOLUTION
# ============================================================

print("\n" + "=" * 70)
print("NODE RESOLUTION")
print("=" * 70)


resolution_counts = Counter()

for edge in relation_edges:

    source = str(edge["source"])
    target = str(edge["target"])

    source_node = nodes.get(source)
    target_node = nodes.get(target)

    source_type = (
        source_node.get("node_type", "UNKNOWN")
        if source_node
        else "MISSING"
    )

    target_type = (
        target_node.get("node_type", "UNKNOWN")
        if target_node
        else "MISSING"
    )

    resolution_counts[
        (source_type, target_type)
    ] += 1


for (source_type, target_type), count in \
        resolution_counts.most_common():

    print(
        f"{source_type:<15} → "
        f"{target_type:<15} : "
        f"{count}"
    )


# ============================================================
# CANONICAL / LOCAL SUMMARY
# ============================================================

canonical_edges = 0
local_edges = 0
mixed_edges = 0
missing_edges = 0

for edge in relation_edges:

    source = str(edge["source"])
    target = str(edge["target"])

    source_node = nodes.get(source)
    target_node = nodes.get(target)

    if source_node is None or target_node is None:

        missing_edges += 1
        continue

    source_type = source_node.get(
        "node_type",
        "UNKNOWN"
    )

    target_type = target_node.get(
        "node_type",
        "UNKNOWN"
    )

    source_canonical = (
        source_type == "ENTITY"
    )

    target_canonical = (
        target_type == "ENTITY"
    )

    source_local = (
        source_type == "MENTION"
    )

    target_local = (
        target_type == "MENTION"
    )

    if source_canonical and target_canonical:

        canonical_edges += 1

    elif source_local and target_local:

        local_edges += 1

    elif (
        source_canonical and target_local
    ) or (
        source_local and target_canonical
    ):

        mixed_edges += 1


print("\nSummary:")

print(
    f"ENTITY → ENTITY : {canonical_edges}"
)

print(
    f"MENTION → MENTION : {local_edges}"
)

print(
    f"ENTITY ↔ MENTION : {mixed_edges}"
)

print(
    f"Missing nodes : {missing_edges}"
)


# ============================================================
# ENTITY TYPE PAIRS
# ============================================================

print("\n" + "=" * 70)
print("ENTITY TYPE PAIRS")
print("=" * 70)

type_pairs = Counter()

for edge in relation_edges:

    props = edge.get(
        "properties",
        {}
    )

    source_type = props.get(
        "source_type",
        "UNKNOWN"
    )

    target_type = props.get(
        "target_type",
        "UNKNOWN"
    )

    relation = edge["relation"]

    type_pairs[
        (
            relation,
            source_type,
            target_type
        )
    ] += 1


for (
    relation,
    source_type,
    target_type
), count in type_pairs.most_common(40):

    print(
        f"{relation:<22}"
        f"{source_type:<20} → "
        f"{target_type:<20}"
        f"{count}"
    )


# ============================================================
# MARGIN STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("MARGIN STATISTICS")
print("=" * 70)

margins = [
    edge["properties"]["margin"]
    for edge in relation_edges
    if edge.get("properties", {}).get(
        "margin"
    ) is not None
]


if margins:

    margins_sorted = sorted(margins)

    def percentile(values, p):

        index = int(
            p * (len(values) - 1)
        )

        return values[index]


    print(
        f"Minimum : {min(margins):.4f}"
    )

    print(
        f"P25     : "
        f"{percentile(margins_sorted, 0.25):.4f}"
    )

    print(
        f"Median  : "
        f"{percentile(margins_sorted, 0.50):.4f}"
    )

    print(
        f"P75     : "
        f"{percentile(margins_sorted, 0.75):.4f}"
    )

    print(
        f"P90     : "
        f"{percentile(margins_sorted, 0.90):.4f}"
    )

    print(
        f"Maximum : {max(margins):.4f}"
    )


# ============================================================
# EXAMPLES PER RELATION
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLES")
print("=" * 70)


examples = defaultdict(list)


for edge in relation_edges:

    relation = edge["relation"]

    if len(examples[relation]) >= 5:

        continue

    props = edge.get(
        "properties",
        {}
    )

    examples[relation].append({

        "source":
            props.get(
                "source_mention",
                edge["source"]
            ),

        "target":
            props.get(
                "target_mention",
                edge["target"]
            ),

        "source_type":
            props.get(
                "source_type",
                "?"
            ),

        "target_type":
            props.get(
                "target_type",
                "?"
            ),

        "margin":
            props.get(
                "margin",
                None
            ),

        "document":
            props.get(
                "document",
                "?"
            ),

        "segment":
            props.get(
                "segment",
                "?"
            ),

        "evidence":
            props.get(
                "evidence_text",
                ""
            )
    })


for relation in relation_counts:

    print(
        f"\n--- {relation} ---"
    )

    for i, example in enumerate(
        examples[relation],
        start=1
    ):

        print(
            f"\nExample {i}"
        )

        print(
            f"Source : "
            f"{example['source']} "
            f"[{example['source_type']}]"
        )

        print(
            f"Target : "
            f"{example['target']} "
            f"[{example['target_type']}]"
        )

        print(
            f"Margin : "
            f"{example['margin']}"
        )

        print(
            f"Document: "
            f"{example['document']}"
        )

        print(
            f"Segment : "
            f"{example['segment']}"
        )

        evidence = example["evidence"]

        if len(evidence) > 500:

            evidence = (
                evidence[:500]
                + "..."
            )

        print(
            f"Evidence: {evidence}"
        )


# ============================================================
# SUSPICIOUS SAME-TYPE RELATIONS
# ============================================================

print("\n" + "=" * 70)
print("SUSPICIOUS SAME-TYPE RELATIONS")
print("=" * 70)


same_type_counts = Counter()

for edge in relation_edges:

    props = edge.get(
        "properties",
        {}
    )

    source_type = props.get(
        "source_type"
    )

    target_type = props.get(
        "target_type"
    )

    if (
        source_type
        and target_type
        and source_type == target_type
    ):

        same_type_counts[
            (
                edge["relation"],
                source_type
            )
        ] += 1


for (
    relation,
    entity_type
), count in same_type_counts.most_common():

    print(
        f"{relation:<22}"
        f"{entity_type:<20}"
        f"{count}"
    )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)