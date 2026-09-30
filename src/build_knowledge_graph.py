import json
import hashlib
from collections import defaultdict, Counter
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "linked_entities_candidates.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "knowledge_graph.json"
)


# ============================================================
# HELPERS
# ============================================================

def make_local_id(document, segment, mention):
    """
    Create a stable ID for an unlinked/local mention.
    """

    raw = f"{document}|{segment}|{mention}"

    digest = hashlib.md5(
        raw.encode("utf-8")
    ).hexdigest()[:12]

    return f"LOCAL_{digest}"


def make_document_id(document):
    """
    Create a stable document node ID.
    """

    return f"DOC_{document}"


def make_canonical_id(candidate_id):
    """
    Create a canonical entity node ID.
    """

    return f"ENT_{candidate_id}"


# ============================================================
# INITIALIZE GRAPH
# ============================================================

nodes = {}
edges = []

document_nodes = set()
canonical_nodes = set()
local_nodes = set()

mentioned_edges = set()
candidate_edges = set()


# ============================================================
# STATISTICS
# ============================================================

total_mentions = 0
linked_mentions = 0
unlinked_mentions = 0
ambiguous_mentions = 0


# ============================================================
# READ ENTITY DATA
# ============================================================

print("=" * 60)
print("BUILDING CYBER THREAT INTELLIGENCE KNOWLEDGE GRAPH")
print("=" * 60)

print("\nInput:")
print(INPUT_FILE)


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        record = json.loads(line)

        document = record["document"]
        segment = record["segment"]
        mention = record["mention"]
        ner_type = record["ner_type"]

        linked = record["linked"]
        conflict = record["conflict"]

        candidates = record["canonical_candidates"]

        total_mentions += 1


        # ====================================================
        # DOCUMENT NODE
        # ====================================================

        document_id = make_document_id(document)

        if document_id not in document_nodes:

            nodes[document_id] = {
                "id": document_id,
                "node_type": "DOCUMENT",
                "document": document
            }

            document_nodes.add(document_id)


        # ====================================================
        # CASE 1: CLEAN CANONICAL LINK
        # ====================================================

        if linked and not conflict and len(candidates) == 1:

            linked_mentions += 1

            candidate = candidates[0]

            canonical_id = make_canonical_id(
                candidate["id"]
            )

            canonical_nodes.add(canonical_id)

            # ----------------------------------------------
            # Canonical entity node
            # ----------------------------------------------

            if canonical_id not in nodes:

                nodes[canonical_id] = {
                    "id": canonical_id,
                    "node_type": "ENTITY",

                    "canonical_id": candidate["id"],
                    "name": candidate["name"],

                    "entity_class": candidate["entity_class"],
                    "entity_type": candidate["entity_type"],

                    "link": candidate["link"]
                }


            # ----------------------------------------------
            # Mention provenance
            # ----------------------------------------------

            edge_key = (
                canonical_id,
                document_id,
                segment,
                mention
            )

            if edge_key not in mentioned_edges:

                edges.append({
                    "source": canonical_id,
                    "target": document_id,

                    "relation": "MENTIONED_IN",

                    "properties": {
                        "mention": mention,
                        "ner_type": ner_type,
                        "segment": segment
                    }
                })

                mentioned_edges.add(edge_key)


        # ====================================================
        # CASE 2: AMBIGUOUS LINK
        # ====================================================

        elif linked and conflict:

            ambiguous_mentions += 1

            # ----------------------------------------------
            # Create a local mention node
            # ----------------------------------------------

            local_id = make_local_id(
                document,
                segment,
                mention
            )

            local_nodes.add(local_id)

            if local_id not in nodes:

                nodes[local_id] = {
                    "id": local_id,
                    "node_type": "MENTION",

                    "mention": mention,
                    "ner_type": ner_type,

                    "document": document,
                    "segment": segment,

                    "status": "AMBIGUOUS"
                }


            # ----------------------------------------------
            # Mention -> Document
            # ----------------------------------------------

            edge_key = (
                local_id,
                document_id
            )

            if edge_key not in mentioned_edges:

                edges.append({
                    "source": local_id,
                    "target": document_id,

                    "relation": "MENTIONED_IN",

                    "properties": {
                        "mention": mention,
                        "segment": segment
                    }
                })

                mentioned_edges.add(edge_key)


            # ----------------------------------------------
            # Mention -> Candidate entities
            # ----------------------------------------------

            for candidate in candidates:

                canonical_id = make_canonical_id(
                    candidate["id"]
                )

                canonical_nodes.add(canonical_id)

                if canonical_id not in nodes:

                    nodes[canonical_id] = {
                        "id": canonical_id,
                        "node_type": "ENTITY",

                        "canonical_id": candidate["id"],
                        "name": candidate["name"],

                        "entity_class": candidate["entity_class"],
                        "entity_type": candidate["entity_type"],

                        "link": candidate["link"]
                    }


                edge_key = (
                    local_id,
                    canonical_id
                )

                if edge_key not in candidate_edges:

                    edges.append({
                        "source": local_id,
                        "target": canonical_id,

                        "relation": "CANDIDATE_FOR",

                        "properties": {
                            "confidence": None
                        }
                    })

                    candidate_edges.add(edge_key)


        # ====================================================
        # CASE 3: UNLINKED
        # ====================================================

        else:

            unlinked_mentions += 1

            local_id = make_local_id(
                document,
                segment,
                mention
            )

            local_nodes.add(local_id)


            # ----------------------------------------------
            # Local entity node
            # ----------------------------------------------

            if local_id not in nodes:

                nodes[local_id] = {
                    "id": local_id,
                    "node_type": "MENTION",

                    "mention": mention,
                    "ner_type": ner_type,

                    "document": document,
                    "segment": segment,

                    "status": "UNLINKED"
                }


            # ----------------------------------------------
            # Mention -> Document
            # ----------------------------------------------

            edge_key = (
                local_id,
                document_id
            )

            if edge_key not in mentioned_edges:

                edges.append({
                    "source": local_id,
                    "target": document_id,

                    "relation": "MENTIONED_IN",

                    "properties": {
                        "mention": mention,
                        "segment": segment
                    }
                })

                mentioned_edges.add(edge_key)


# ============================================================
# GRAPH OBJECT
# ============================================================

graph = {
    "metadata": {
        "name": "AnnoCTR Cyber Threat Intelligence Knowledge Graph",
        "version": "1.0",
        "source": "AnnoCTR",
        "description": (
            "Entity-centric graph constructed from AnnoCTR "
            "entity mentions and entity linking annotations."
        )
    },

    "nodes": list(nodes.values()),

    "edges": edges
}


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
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# STATISTICS
# ============================================================

node_type_counts = Counter(
    node["node_type"]
    for node in nodes.values()
)

edge_type_counts = Counter(
    edge["relation"]
    for edge in edges
)


print("\n" + "=" * 60)
print("GRAPH STATISTICS")
print("=" * 60)

print(f"\nEntity mentions processed : {total_mentions}")

print(f"Cleanly linked           : {linked_mentions}")

print(f"Unlinked                 : {unlinked_mentions}")

print(f"Ambiguous                : {ambiguous_mentions}")


print("\nNodes:")

for node_type, count in node_type_counts.items():

    print(
        f"  {node_type:15s}: {count}"
    )


print("\nEdges:")

for edge_type, count in edge_type_counts.items():

    print(
        f"  {edge_type:15s}: {count}"
    )


print("\nTotal nodes:", len(nodes))

print("Total edges:", len(edges))


print("\nOutput:")
print(OUTPUT_FILE)

print("\nDone.")