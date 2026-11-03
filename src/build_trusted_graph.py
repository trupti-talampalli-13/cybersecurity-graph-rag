"""Build a layered, provenance-aware CTI knowledge graph with a trust level on every assertion.

The original graph (data/cti_knowledge_graph.json) and the automatic relations (data/relations.jsonl) are
left unchanged. This writes a separate, clearly named copy in which every edge belongs to one layer:

  layer "mention"            entity/mention -> document   (what the report says, with source and date)
  layer "candidate_link"     ambiguous mention -> candidate entity   (unresolved entity links)
  layer "relation"           automatic positive assertion, with trust_level:
      trusted    actor-headed USES/TARGETS or USES_TECHNIQUE, high/medium confidence, no integrity flag
      candidate  needs verification: malware/tool-headed USES/TARGETS, EXPLOITS, or an integrity flag
      low        low confidence (hedged wording or context-borrowed subject)
  layer "negative_assertion" a sentence that says the relation does NOT hold ("never used", "did not use").
                             Kept in its own layer, never as a plain relation edge, so a traversal that follows
                             relation edges cannot mistake it for a positive claim. Always `candidate`.

Every assertion carries its context: subject, predicate, object, polarity, publication date, best-effort scope
(campaign and dates named in the sentence), evidence sentence, source document, certainty (asserted/hedged),
extraction confidence, trust level and verification status (`unverified` until the semantic verifier sets it).
No assertion is human-verified (`human_verified` is always false). The trust policy follows a human review of
66 relations from model-tagged unseen reports (README, "Human review of round 2"): actor-headed relations
were right 75-80% of the time, malware-headed `USES`/`TARGETS` far less often.

Writes: data/cti_knowledge_graph_trusted.json   full layered graph
        data/relations_trusted.jsonl            only the `trusted` positive relations, flat, for retrieval
        data/assertions.jsonl                   every assertion (positive and negative) with its full context

Run from the cti_graph_rag directory:  python src/build_trusted_graph.py
"""
import json
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
RELATION_TYPES = {"USES", "USES_TECHNIQUE", "EXPLOITS", "TARGETS"}
LAYERS = ("mention", "candidate_link", "relation", "negative_assertion")
ASSERTION_LAYERS = ("relation", "negative_assertion")
CONFIDENCES = {"high", "medium", "low"}
CERTAINTIES = {"asserted", "hedged"}
POLARITIES = {"positive", "negative"}
TRUST_LEVELS = ("trusted", "candidate", "low")
# "unverified" until the semantic verifier (claim-level evidence verification) sets one of the others
VERIFICATION_STATUSES = ("unverified", "supported", "contradicted", "insufficient_evidence")
REQUIRED_RELATION_PROPS = ("document", "segment", "source", "publication_date", "evidence_text", "trigger",
                           "direction", "confidence", "head_mention", "tail_mention", "head_type", "tail_type",
                           "polarity", "certainty", "scope", "verification_status")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def assign_trust(relation, props, head_node, tail_node, schema):
    """Return (trust_level, reasons) for one assertion edge."""
    reasons = []
    allowed = schema.get(relation, {})
    if head_node.get("entity_type") not in allowed.get("head", []) or props["head_type"] not in allowed.get("head", []):
        reasons.append("head_type_mismatch")
    if tail_node.get("entity_type") not in allowed.get("tail", []) or props["tail_type"] not in allowed.get("tail", []):
        reasons.append("tail_type_mismatch")
    if props.get("polarity") == "negative":
        return "candidate", ["negative_polarity"] + reasons     # negated claims are never trusted by default
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
        p = e["properties"]
        if e["source"] not in known or e["target"] not in known:
            problems.append(f"dangling edge {e['source']} -> {e['target']}")
        if e["source"] == e["target"]:
            problems.append(f"self loop {e['source']}")
        if p.get("layer") not in LAYERS:
            problems.append(f"edge without layer {e['source']} -> {e['target']}")
        if e["relation"] in RELATION_TYPES:
            missing = [k for k in REQUIRED_RELATION_PROPS if p.get(k) in (None, "")]
            if missing:
                problems.append(f"relation missing {missing}: {e['source']} {e['relation']} {e['target']}")
            if p.get("layer") not in ASSERTION_LAYERS:
                problems.append(f"assertion edge in layer {p.get('layer')!r}")
            if not DATE_RE.match(str(p.get("publication_date", ""))):
                problems.append(f"bad date {p.get('publication_date')!r}")
            if p.get("confidence") not in CONFIDENCES:
                problems.append(f"bad confidence {p.get('confidence')!r}")
            if p.get("certainty") not in CERTAINTIES:
                problems.append(f"bad certainty {p.get('certainty')!r}")
            if p.get("polarity") not in POLARITIES:
                problems.append(f"bad polarity {p.get('polarity')!r}")
            if (p.get("polarity") == "negative") != (p.get("layer") == "negative_assertion"):
                problems.append("polarity and layer disagree (negative claims belong only in negative_assertion)")
            if p.get("trust_level") not in TRUST_LEVELS:
                problems.append(f"bad trust level {p.get('trust_level')!r}")
            if p.get("verification_status") not in VERIFICATION_STATUSES:
                problems.append(f"bad verification status {p.get('verification_status')!r}")
            if not isinstance(p.get("scope"), dict) or not {"campaign", "dates"} <= set(p["scope"]):
                problems.append("scope must be an object with 'campaign' and 'dates'")
            if p.get("human_verified") is not False:
                problems.append("human_verified must be false")
    return problems


def assertion_row(assertion_id, edge, nodes):
    """Flat record for data/assertions.jsonl, using the plan's field names."""
    p, s, o = edge["properties"], nodes[edge["source"]], nodes[edge["target"]]
    return {
        "assertion_id": assertion_id,
        "subject": s.get("name"), "subject_id": s["id"], "subject_type": s.get("entity_type"),
        "predicate": edge["relation"],
        "object": o.get("name"), "object_id": o["id"], "object_type": o.get("entity_type"),
        "polarity": p["polarity"],
        "publication_date": p["publication_date"],
        "scope": p["scope"],
        "evidence_sentence": p["evidence_text"],
        "source_document": p["document"], "source_vendor": p["source"], "segment_id": p["segment"],
        "subject_mention": p["head_mention"], "object_mention": p["tail_mention"],
        "trigger": p["trigger"], "direction": p["direction"],
        "certainty": p["certainty"], "confidence": p["confidence"],
        "trust_level": p["trust_level"], "trust_reasons": p["trust_reasons"],
        "verification_status": p["verification_status"], "human_verified": p["human_verified"],
        "layer": p["layer"], "method": p["method"],
    }


def main():
    graph = json.load(open(DATA / "cti_knowledge_graph.json", encoding="utf-8"))
    nodes = {n["id"]: n for n in graph["nodes"]}
    schema = graph["metadata"]["relation_schema"]
    records = [json.loads(line) for line in open(DATA / "relations.jsonl", encoding="utf-8")]
    keys = Counter((e["source"], e["relation"], e["target"], e["properties"].get("document"),
                    e["properties"].get("segment")) for e in graph["edges"] if e["relation"] in RELATION_TYPES)

    # negative assertions come from relations.jsonl (the plain graph deliberately has no edges for them)
    negative_edges = []
    for r in records:
        if r["polarity"] == "negative":
            props = {k: v for k, v in r.items() if k not in ("head", "relation", "tail")}
            negative_edges.append({"source": r["head"], "target": r["tail"], "relation": r["relation"],
                                   "properties": props})

    counts = Counter()
    for e in graph["edges"] + negative_edges:
        p = e["properties"]
        if e["relation"] == "MENTIONED_IN":
            p["layer"] = "mention"
        elif e["relation"] == "CANDIDATE_FOR":
            p["layer"] = "candidate_link"
        else:
            level, reasons = assign_trust(e["relation"], p, nodes[e["source"]], nodes[e["target"]], schema)
            if keys[(e["source"], e["relation"], e["target"], p["document"], p["segment"])] > 1 \
                    and p["polarity"] == "positive":
                reasons.append("duplicate_instance")
                level = "candidate" if level == "trusted" else level
            p.update({"layer": "negative_assertion" if p["polarity"] == "negative" else "relation",
                      "trust_level": level, "trust_reasons": reasons, "human_verified": False})
            counts[(e["relation"], p["polarity"], level)] += 1
    graph["edges"] = graph["edges"] + negative_edges

    assertion_edges = [e for e in graph["edges"] if e["properties"].get("layer") in ASSERTION_LAYERS]
    assertion_edges.sort(key=lambda e: (e["properties"]["document"], e["properties"]["segment"], e["relation"],
                                        e["source"], e["target"], e["properties"]["polarity"]))
    assertions = [assertion_row(f"A{i:05d}", e, nodes) for i, e in enumerate(assertion_edges, 1)]
    positive_levels = Counter(a["trust_level"] for a in assertions if a["polarity"] == "positive")

    graph["metadata"].update({
        "name": "CTI Knowledge Graph (layered, provenance-aware, with trust levels)",
        "version": "4.0",
        "derived_from": "data/cti_knowledge_graph.json (unchanged) and data/relations.jsonl",
        "layers": {
            "mention": "entity or mention node -> document, with source and publication date",
            "candidate_link": "ambiguous mention -> candidate canonical entity (link not resolved)",
            "relation": "automatic positive assertion; see trust_level",
            "negative_assertion": "a sentence stating that the relation does NOT hold; never a plain relation edge",
        },
        "assertion_context": {
            "fields": ["subject", "predicate", "object", "polarity", "publication_date", "scope",
                       "evidence_sentence", "source_document", "certainty", "confidence", "trust_level",
                       "verification_status"],
            "polarity": "positive, or negative when a negation cue ('never', 'not', \"n't\") directly precedes the trigger",
            "certainty": "asserted, or hedged ('likely', 'potentially', 'allegedly'); hedged assertions also have low confidence",
            "scope": "best effort from the sentence only: a named campaign (regex) and the dates it mentions; "
                     "not a campaign model, and campaign names may be malware or actor names",
            "verification_status": "unverified until the semantic verifier sets supported, contradicted or "
                                   "insufficient_evidence",
        },
        "trust_policy": {
            "trusted": "USES/TARGETS with a threat-actor head, or USES_TECHNIQUE; confidence high or medium; "
                       "node and mention types match the schema; not a duplicate instance",
            "candidate": "needs verification: malware/tool-headed USES/TARGETS, EXPLOITS, type mismatch, duplicate, "
                         "or any negative assertion",
            "low": "low confidence (hedged wording or subject borrowed from a nearby sentence)",
        },
        "trust_note": "No assertion is human-verified. Trust levels come from a policy based on a human review of 66 "
                      "relations from model-tagged unseen reports: actor-headed relations were right 75-80% of the "
                      "time, malware-headed USES/TARGETS far less often. On the gold-tagged reports used here, "
                      "malware-headed relations looked accurate in a provisional single-reviewer audit, so "
                      "'candidate' is conservative for this graph.",
        "relation_trust_counts": dict(positive_levels),
        "negative_assertion_count": sum(1 for a in assertions if a["polarity"] == "negative"),
    })
    problems = validate(graph)
    if problems:
        raise SystemExit("validation failed:\n" + "\n".join(problems[:20]))

    with open(DATA / "cti_knowledge_graph_trusted.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    trusted = [a for a in assertions if a["polarity"] == "positive" and a["trust_level"] == "trusted"]
    with open(DATA / "relations_trusted.jsonl", "w", encoding="utf-8") as f:
        for a in trusted:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(DATA / "assertions.jsonl", "w", encoding="utf-8") as f:
        for a in assertions:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    print("Validation passed.")
    print(f"Positive relations by trust level: {dict(positive_levels)}; "
          f"negative assertions: {graph['metadata']['negative_assertion_count']}")
    for (rel, pol, level), n in sorted(counts.items()):
        print(f"  {rel:15s} {pol:9s} {level:10s} {n}")
    print(f"Wrote data/cti_knowledge_graph_trusted.json ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges), "
          f"data/relations_trusted.jsonl ({len(trusted)} trusted) and data/assertions.jsonl ({len(assertions)} assertions)")


if __name__ == "__main__":
    main()
