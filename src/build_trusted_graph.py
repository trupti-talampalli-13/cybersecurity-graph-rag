"""Build a layered CTI knowledge graph with an explicit trust level on every relation.

The original graph (data/cti_knowledge_graph.json) and the automatic relations
(data/relations.jsonl) are left unchanged. This writes a separate, clearly named copy in which every
edge belongs to one layer and every relation carries a trust level:

  layer "mention"        entity/mention -> document        (what the report says, with source and date)
  layer "candidate_link" ambiguous mention -> candidate entity   (unresolved entity links)
  layer "relation"       automatically extracted relation, with trust_level:
      trusted    actor-headed USES/TARGETS or USES_TECHNIQUE, high/medium confidence, no integrity flag
      candidate  needs verification: malware/tool-headed USES/TARGETS, EXPLOITS, or an integrity flag
      low        low confidence (hedged wording or context-borrowed subject)

No relation is human-verified (`human_verified` is always false). The policy follows a human review of
66 relations from model-tagged unseen reports (README, "Human review of round 2"): actor-headed relations
were right 75-80% of the time, malware-headed `USES`/`TARGETS` far less often.

Writes: data/cti_knowledge_graph_trusted.json   full layered graph
        data/relations_trusted.jsonl            only the `trusted` relations, flat, for retrieval

Run from the cti_graph_rag directory:  python src/build_trusted_graph.py
"""
import json
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
RELATION_TYPES = {"USES", "USES_TECHNIQUE", "EXPLOITS", "TARGETS"}
CONFIDENCES = {"high", "medium", "low"}
TRUST_LEVELS = ("trusted", "candidate", "low")
REQUIRED_RELATION_PROPS = ("document", "segment", "source", "publication_date", "evidence_text", "trigger",
                           "direction", "confidence", "head_mention", "tail_mention", "head_type", "tail_type")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def assign_trust(relation, props, head_node, tail_node, schema):
    """Return (trust_level, reasons) for one relation edge."""
    reasons = []
    allowed = schema.get(relation, {})
    if head_node.get("entity_type") not in allowed.get("head", []) or props["head_type"] not in allowed.get("head", []):
        reasons.append("head_type_mismatch")
    if tail_node.get("entity_type") not in allowed.get("tail", []) or props["tail_type"] not in allowed.get("tail", []):
        reasons.append("tail_type_mismatch")
    if props["confidence"] == "low":
        return "low", ["low_confidence"] + reasons
    if relation == "EXPLOITS":
        reasons.append("exploits_rarely_extracted")
    elif relation in ("USES", "TARGETS") and props["head_type"] != "GROUP":
        reasons.append("malware_or_tool_headed")
    return ("candidate" if reasons else "trusted"), reasons


def validate(graph):
    """Structural checks. Returns a list of problems (empty when the graph is sound)."""
    problems, ids = [], [n["id"] for n in graph["nodes"]]
    if len(ids) != len(set(ids)):
        problems.append("duplicate node ids")
    known = set(ids)
    for e in graph["edges"]:
        if e["source"] not in known or e["target"] not in known:
            problems.append(f"dangling edge {e['source']} -> {e['target']}")
        if e["source"] == e["target"]:
            problems.append(f"self loop {e['source']}")
        if e["properties"].get("layer") not in ("mention", "candidate_link", "relation"):
            problems.append(f"edge without layer {e['source']} -> {e['target']}")
        if e["relation"] in RELATION_TYPES:
            p = e["properties"]
            missing = [k for k in REQUIRED_RELATION_PROPS if p.get(k) in (None, "")]
            if missing:
                problems.append(f"relation missing {missing}: {e['source']} {e['relation']} {e['target']}")
            if not DATE_RE.match(str(p.get("publication_date", ""))):
                problems.append(f"bad date {p.get('publication_date')!r}")
            if p.get("confidence") not in CONFIDENCES:
                problems.append(f"bad confidence {p.get('confidence')!r}")
            if p.get("trust_level") not in TRUST_LEVELS:
                problems.append(f"bad trust level {p.get('trust_level')!r}")
            if p.get("human_verified") is not False:
                problems.append("human_verified must be false")
    return problems


def main():
    graph = json.load(open(DATA / "cti_knowledge_graph.json", encoding="utf-8"))
    nodes = {n["id"]: n for n in graph["nodes"]}
    schema = graph["metadata"]["relation_schema"]
    keys = Counter((e["source"], e["relation"], e["target"], e["properties"].get("document"),
                    e["properties"].get("segment")) for e in graph["edges"] if e["relation"] in RELATION_TYPES)

    trusted_rows, counts = [], Counter()
    for e in graph["edges"]:
        p = e["properties"]
        if e["relation"] == "MENTIONED_IN":
            p["layer"] = "mention"
        elif e["relation"] == "CANDIDATE_FOR":
            p["layer"] = "candidate_link"
        else:
            level, reasons = assign_trust(e["relation"], p, nodes[e["source"]], nodes[e["target"]], schema)
            if keys[(e["source"], e["relation"], e["target"], p["document"], p["segment"])] > 1:
                reasons.append("duplicate_instance")
                level = "candidate" if level == "trusted" else level
            p.update({"layer": "relation", "trust_level": level, "trust_reasons": reasons, "human_verified": False})
            counts[(e["relation"], level)] += 1
            if level == "trusted":
                trusted_rows.append({"head": e["source"], "head_name": nodes[e["source"]].get("name"),
                                     "relation": e["relation"], "tail": e["target"],
                                     "tail_name": nodes[e["target"]].get("name"), **p})

    levels = Counter(level for (_, level) in counts.elements())
    graph["metadata"].update({
        "name": "CTI Knowledge Graph (layered, with trust levels)",
        "version": "3.0",
        "derived_from": "data/cti_knowledge_graph.json (unchanged) and data/relations.jsonl",
        "layers": {
            "mention": "entity or mention node -> document, with source and publication date",
            "candidate_link": "ambiguous mention -> candidate canonical entity (link not resolved)",
            "relation": "automatically extracted relation; see trust_level",
        },
        "trust_policy": {
            "trusted": "USES/TARGETS with a threat-actor head, or USES_TECHNIQUE; confidence high or medium; "
                       "node and mention types match the schema; not a duplicate instance",
            "candidate": "needs verification: malware/tool-headed USES/TARGETS, EXPLOITS, type mismatch, duplicate",
            "low": "low confidence (hedged wording or subject borrowed from a nearby sentence)",
        },
        "trust_note": "No relation is human-verified. Trust levels come from a policy based on a human review of 66 "
                      "relations from model-tagged unseen reports: actor-headed relations were right 75-80% of the "
                      "time, malware-headed USES/TARGETS far less often. On the gold-tagged reports used here, "
                      "malware-headed relations looked accurate in a provisional single-reviewer audit, so "
                      "'candidate' is conservative for this graph.",
        "relation_trust_counts": dict(levels),
    })
    problems = validate(graph)
    if problems:
        raise SystemExit("validation failed:\n" + "\n".join(problems[:20]))

    with open(DATA / "cti_knowledge_graph_trusted.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    with open(DATA / "relations_trusted.jsonl", "w", encoding="utf-8") as f:
        for row in trusted_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Validation passed.")
    print(f"Relations by trust level: {dict(levels)}")
    for (rel, level), n in sorted(counts.items()):
        print(f"  {rel:15s} {level:10s} {n}")
    print(f"Wrote data/cti_knowledge_graph_trusted.json ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges) "
          f"and data/relations_trusted.jsonl ({len(trusted_rows)} trusted relations)")


if __name__ == "__main__":
    main()
