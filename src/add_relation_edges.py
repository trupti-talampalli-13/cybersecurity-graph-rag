import json
import hashlib
from collections import Counter

GRAPH_FILE = "data/knowledge_graph.json"
LINK_FILE = "data/linked_entities_candidates.jsonl"
PRED_FILE = "data/annoctr_relation_predictions.jsonl"
OUTPUT_FILE = "data/semantic_knowledge_graph.json"

MARGIN_THRESHOLD = 0.70


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():

                records.append(
                    json.loads(line)
                )

    return records


def local_node_id(
    document,
    segment,
    mention,
    entity_type
):

    raw = (
        f"{document}|"
        f"{segment}|"
        f"{mention}|"
        f"{entity_type}"
    )

    return (
        "LOCAL_"
        + hashlib.md5(
            raw.encode("utf-8")
        ).hexdigest()[:16]
    )


# ============================================================
# LOAD EXISTING GRAPH
# ============================================================

print("=" * 70)
print("BUILDING SEMANTIC CTI KNOWLEDGE GRAPH")
print("=" * 70)

print("\nLoading existing knowledge graph...")

with open(
    GRAPH_FILE,
    "r",
    encoding="utf-8"
) as f:

    graph = json.load(f)


print(
    f"Existing nodes: "
    f"{len(graph['nodes'])}"
)

print(
    f"Existing edges: "
    f"{len(graph['edges'])}"
)


# ============================================================
# EXISTING NODE IDS
# ============================================================

node_ids = set(
    str(node["id"])
    for node in graph["nodes"]
)


# ============================================================
# LOAD ENTITY LINKING
# ============================================================

print(
    "\nLoading entity-linking candidates..."
)

link_records = load_jsonl(
    LINK_FILE
)

print(
    f"Linking records: "
    f"{len(link_records)}"
)


# ============================================================
# BUILD UNAMBIGUOUS ENTITY MAPPING
# ============================================================

unique_mapping = {}

for record in link_records:

    candidates = record.get(
        "canonical_candidates",
        []
    )

    if (
        record.get("linked") is True
        and record.get("conflict") is False
        and len(candidates) == 1
    ):

        candidate = candidates[0]

        key = (
            record["document"],
            record["segment"],
            record["normalized_mention"],
            record["ner_type"]
        )

        # IMPORTANT:
        # The linker stores the canonical ID as an integer.
        # The KG stores canonical nodes as ENT_<ID>.
        canonical_id = (
            f"ENT_{candidate['id']}"
        )

        unique_mapping[key] = canonical_id


print(
    f"Unique unambiguous mappings: "
    f"{len(unique_mapping)}"
)


# ============================================================
# VERIFY MAPPINGS AGAINST GRAPH
# ============================================================

valid_mappings = {
    key: node_id
    for key, node_id in unique_mapping.items()
    if node_id in node_ids
}

invalid_mapping_count = (
    len(unique_mapping)
    - len(valid_mappings)
)

unique_mapping = valid_mappings


print(
    f"Mappings found in existing KG: "
    f"{len(unique_mapping)}"
)

print(
    f"Mappings missing from existing KG: "
    f"{invalid_mapping_count}"
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print(
    "\nLoading SVM relation predictions..."
)

predictions = load_jsonl(
    PRED_FILE
)

print(
    f"Predictions loaded: "
    f"{len(predictions)}"
)


# ============================================================
# EXISTING EDGE KEYS
# ============================================================

edge_keys = set()

for edge in graph["edges"]:

    properties = edge.get(
        "properties",
        {}
    )

    key = (
        str(edge.get("source")),
        str(edge.get("target")),
        edge.get("relation"),
        properties.get("document"),
        properties.get("segment")
    )

    edge_keys.add(key)


# ============================================================
# STATISTICS
# ============================================================

low_margin = 0
duplicate_edges = 0
relation_edges_added = 0

canonical_source_links = 0
canonical_target_links = 0

local_source_mentions = 0
local_target_mentions = 0

relation_counter = Counter()


# ============================================================
# PROCESS RELATION PREDICTIONS
# ============================================================

for pred in predictions:

    margin = pred.get(
        "margin",
        0.0
    )

    # --------------------------------------------------------
    # CONFIDENCE FILTER
    # --------------------------------------------------------

    if margin < MARGIN_THRESHOLD:

        low_margin += 1

        continue


    document = pred["document"]
    segment = pred["segment"]

    source_mention = pred[
        "source_mention"
    ]

    target_mention = pred[
        "target_mention"
    ]

    source_type = pred[
        "source_type"
    ]

    target_type = pred[
        "target_type"
    ]

    relation = pred[
        "predicted_relation"
    ]


    # --------------------------------------------------------
    # NORMALIZED LOOKUP
    # --------------------------------------------------------

    source_normalized = (
        source_mention
        .strip()
        .lower()
    )

    target_normalized = (
        target_mention
        .strip()
        .lower()
    )


    source_key = (
        document,
        segment,
        source_normalized,
        source_type
    )

    target_key = (
        document,
        segment,
        target_normalized,
        target_type
    )


    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    source_id = unique_mapping.get(
        source_key
    )

    if source_id is not None:

        canonical_source_links += 1

    else:

        source_id = local_node_id(
            document,
            segment,
            source_mention,
            source_type
        )

        local_source_mentions += 1

        if source_id not in node_ids:

            graph["nodes"].append({

                "id": source_id,

                "node_type": "MENTION",

                "document": document,

                "segment": segment,

                "mention": source_mention,

                "entity_type": source_type
            })

            node_ids.add(
                source_id
            )


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    target_id = unique_mapping.get(
        target_key
    )

    if target_id is not None:

        canonical_target_links += 1

    else:

        target_id = local_node_id(
            document,
            segment,
            target_mention,
            target_type
        )

        local_target_mentions += 1

        if target_id not in node_ids:

            graph["nodes"].append({

                "id": target_id,

                "node_type": "MENTION",

                "document": document,

                "segment": segment,

                "mention": target_mention,

                "entity_type": target_type
            })

            node_ids.add(
                target_id
            )


    # --------------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------------

    edge_key = (
        str(source_id),
        str(target_id),
        relation,
        document,
        segment
    )

    if edge_key in edge_keys:

        duplicate_edges += 1

        continue

    edge_keys.add(
        edge_key
    )


    # --------------------------------------------------------
    # ADD RELATION
    # --------------------------------------------------------

    graph["edges"].append({

        "source": source_id,

        "target": target_id,

        "relation": relation,

        "properties": {

            "document": document,

            "segment": segment,

            "evidence_text": pred.get(
                "text",
                pred.get(
                    "evidence_text",
                    ""
                )
            ),

            "source_mention":
                source_mention,

            "target_mention":
                target_mention,

            "source_type":
                source_type,

            "target_type":
                target_type,

            "decision_score":
                pred.get(
                    "decision_score"
                ),

            "margin":
                margin,

            "margin_threshold":
                MARGIN_THRESHOLD,

            "method":
                "AZERG_T4_LinearSVC"
        }
    })


    relation_edges_added += 1

    relation_counter[
        relation
    ] += 1


# ============================================================
# RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RELATION EXTRACTION RESULTS"
)

print(
    "=" * 70
)


print(
    f"Predictions loaded       : "
    f"{len(predictions)}"
)

print(
    f"Low-margin predictions   : "
    f"{low_margin}"
)

print(
    f"Duplicate edges          : "
    f"{duplicate_edges}"
)

print(
    f"Relation edges added     : "
    f"{relation_edges_added}"
)


print(
    f"Canonical source links   : "
    f"{canonical_source_links}"
)

print(
    f"Canonical target links   : "
    f"{canonical_target_links}"
)

print(
    f"Local source mentions    : "
    f"{local_source_mentions}"
)

print(
    f"Local target mentions    : "
    f"{local_target_mentions}"
)


print(
    "\nRelation edges added:"
)

for relation, count in (
    relation_counter
    .most_common()
):

    print(
        f"{relation:<25}"
        f": {count}"
    )


# ============================================================
# FINAL GRAPH STATISTICS
# ============================================================

node_counts = Counter(
    node.get(
        "node_type",
        "UNKNOWN"
    )
    for node in graph["nodes"]
)

edge_counts = Counter(
    edge.get(
        "relation",
        "UNKNOWN"
    )
    for edge in graph["edges"]
)


print(
    "\n"
    + "=" * 70
)

print(
    "FINAL SEMANTIC KNOWLEDGE GRAPH"
)

print(
    "=" * 70
)


print(
    f"Total nodes : "
    f"{len(graph['nodes'])}"
)

print(
    f"Total edges : "
    f"{len(graph['edges'])}"
)


print(
    "\nNode types:"
)

for node_type, count in (
    node_counts.most_common()
):

    print(
        f"{node_type:<20}"
        f": {count}"
    )


print(
    "\nEdge types:"
)

for edge_type, count in (
    edge_counts.most_common()
):

    print(
        f"{edge_type:<25}"
        f": {count}"
    )


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        graph,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    "\n"
    + "=" * 70
)

print(
    "DONE"
)

print(
    "=" * 70
)

print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)