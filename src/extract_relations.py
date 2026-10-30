"""Rule-based extraction of USES, USES_TECHNIQUE, EXPLOITS and TARGETS relations.

For every segment, entity pairs are proposed from the gold/regex entity spans, kept only
if their types fit the relation schema and a trigger phrase links them in the text.
Endpoints are resolved to canonical graph node ids so relations connect across reports.

Reads:  data/processed_segments.jsonl, data/linked_entities_candidates.jsonl, data/knowledge_graph.json
Writes: data/relations.jsonl            one record per extracted relation instance
        data/cti_knowledge_graph.json   base graph + resolved entity nodes + relation edges

Run from the cti_graph_rag directory:  python src/extract_relations.py
"""
import json
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

ACTORS = {"GROUP"}
CARRIERS = {"GROUP", "MALWARE", "TOOL"}      # things that can use / exploit
VICTIMS = {"ORG", "SECTOR", "LOC"}

# relation -> (allowed head types, allowed tail types)
SCHEMA = {
    "USES": ({"GROUP", "MALWARE"}, {"MALWARE", "TOOL"}),
    "USES_TECHNIQUE": (CARRIERS, {"TECHNIQUE", "TACTIC"}),
    "EXPLOITS": (CARRIERS, {"CVE"}),
    "TARGETS": ({"GROUP", "MALWARE"}, VICTIMS),
}

NOT_PASSIVE = r"(?!\s+by\b)"   # "distributed by X" is passive, not "X distributed ..."
ACTIVE = {
    "USES": r"\b(us(?:e|es|ed|ing)|utili[sz]\w*|employ\w*|deploy\w*|drop(?:s|ped|ping)?|deliver\w*|"
            r"distribut\w*|install\w*|download\w*|load(?:s|ed|ing)?|fetch\w*|leverag\w*|"
            r"incorporat\w*|launch\w*|abus\w*)\b" + NOT_PASSIVE,
    "USES_TECHNIQUE": r"\b(us(?:e|es|ed|ing)|employ\w*|perform\w*|leverag\w*|achiev\w*|conduct\w*|"
                      r"implement\w*|abus\w*|via|through|utili[sz]\w*|execut\w*)\b" + NOT_PASSIVE,
    "EXPLOITS": r"\b(exploit\w*|leverag\w*|abus\w*|weaponi[sz]\w*|take[sn]? advantage of)\b" + NOT_PASSIVE,
    "TARGETS": r"\b(target\w*|attack\w*|hit(?:s|ting)?|victimi[sz]\w*|compromis\w*|infect\w*|"
               r"aim(?:s|ed|ing)? at|against|focus\w* on|go(?:es|ing)? after|breach\w*)\b" + NOT_PASSIVE,
}
# passive: "<tail> ... <participle> by <head>"  (reverses reading direction)
PASSIVE = {
    "USES": r"\b(us|utili[sz]|employ|deploy|drop|deliver|distribut|install|download|load|launch|execut)ed\s+by\b",
    "USES_TECHNIQUE": r"\b(us|employ|perform|leverag|conduct|implement|utili[sz])ed\s+by\b",
    "EXPLOITS": r"\b(exploit|leverag|abus|weaponi[sz])ed\s+by\b",
    "TARGETS": r"\b(target|attack|hit|victimi[sz]|compromis|infect|affect)ed\s+by\b",
}
ACTIVE_RE = {r: re.compile(p, re.I) for r, p in ACTIVE.items()}
PASSIVE_RE = {r: re.compile(p, re.I) for r, p in PASSIVE.items()}
# max characters between the two mentions, per relation (and for passive phrasing)
MAX_GAP = {"USES": 120, "USES_TECHNIQUE": 70, "EXPLOITS": 160, "TARGETS": 100}
MAX_PASSIVE_GAP = 60
MAX_TAIL_WORDS = {"USES_TECHNIQUE": 4}
STRONG_TARGET = re.compile(r"target\w*|aim\w*|against|focus\w* on|go(?:es|ing)? after|victimi[sz]\w*", re.I)


def norm(text):
    text = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"\s*-\s*", "-", text)


def load_resolution():
    """(document, segment, mention) -> canonical ENT_ id when the mention links cleanly."""
    res = {}
    with open(DATA / "linked_entities_candidates.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            c = r["canonical_candidates"]
            if r["linked"] and not r["conflict"] and len(c) == 1:
                res[(r["document"], r["segment"], r["mention"])] = (f"ENT_{c[0]['id']}", c[0])
    return res


def resolve(doc, seg, ent, resolution):
    hit = resolution.get((doc, seg, ent["text"]))
    if hit:
        return hit[0], {"name": hit[1]["name"], "entity_type": hit[1]["entity_type"],
                        "link": hit[1]["link"], "canonical_id": hit[1]["id"]}
    key = norm(ent["text"])
    return f"NORM_{ent['type']}_{key}", {"name": ent["text"], "entity_type": ent["type"],
                                         "canonical_id": None, "link": None}


def find_relations(seg, resolution):
    text, ents = seg["text"], [e for e in seg["entities"] if e["char_start"] is not None]
    ents.sort(key=lambda e: e["char_start"])
    out = []
    for rel, (head_types, tail_types) in SCHEMA.items():
        heads = [e for e in ents if e["type"] in head_types]
        tails = [e for e in ents if e["type"] in tail_types]
        for h in heads:
            for t in tails:
                if h is t:
                    continue
                if h["char_end"] <= t["char_start"]:          # head ... tail
                    between = text[h["char_end"]:t["char_start"]]
                    direction = "active"
                    trig = ACTIVE_RE[rel].search(between)
                elif t["char_end"] <= h["char_start"]:        # tail ... head
                    between = text[t["char_end"]:h["char_start"]]
                    direction = "passive"
                    trig = PASSIVE_RE[rel].search(between)
                else:
                    continue
                if not trig or len(between) > (MAX_PASSIVE_GAP if direction == "passive" else MAX_GAP[rel]):
                    continue
                if len(t["text"].split()) > MAX_TAIL_WORDS.get(rel, 99):
                    continue
                if rel == "TARGETS" and t["type"] == "LOC" and not STRONG_TARGET.match(trig.group(0)):
                    continue
                inner = [e for e in ents if e not in (h, t)
                         and min(h["char_end"], t["char_end"]) <= e["char_start"]
                         and e["char_end"] <= max(h["char_start"], t["char_start"])
                         and (e["type"] in head_types or e["type"] in tail_types)]
                conf = "high" if not inner and len(between) <= 40 else "medium"
                out.append((rel, h, t, trig.group(0).lower(), direction, conf))
    return out


def cross_segment_exploits(seg, recent_carriers):
    """EXPLOITS where the sentence names a CVE but no actor/malware: borrow the single
    carrier mentioned in the previous two segments of the same report (low confidence)."""
    cves = [e for e in seg["entities"] if e["type"] == "CVE" and e["char_start"] is not None]
    carriers = {(e["text"], e["type"]): e for e in recent_carriers}
    if any(e["type"] in CARRIERS for e in seg["entities"]) or len(carriers) != 1:
        return []
    carrier = next(iter(carriers.values()))
    m = ACTIVE_RE["EXPLOITS"].search(seg["text"])
    return [("EXPLOITS", carrier, c, m.group(0).lower(), "context", "low") for c in cves] if m else []


def main():
    resolution = load_resolution()
    recent = {}  # report_id -> [(segment_id, carrier entities)]
    base = json.load(open(DATA / "knowledge_graph.json", encoding="utf-8"))
    nodes = {n["id"]: n for n in base["nodes"]}
    edges = list(base["edges"])
    stats, seen, records = Counter(), set(), []

    with open(DATA / "processed_segments.jsonl", encoding="utf-8") as f:
        for line in f:
            seg = json.loads(line)
            doc, sid = seg["report_id"], seg["segment_id"]
            found = find_relations(seg, resolution)
            hist = recent.setdefault(doc, [])
            prev = [e for sid0, es in hist if 0 < sid - sid0 <= 2 for e in es]
            found += cross_segment_exploits(seg, prev)
            hist.append((sid, [e for e in seg["entities"] if e["type"] in CARRIERS]))
            for rel, h, t, trig, direction, conf in found:
                hid, hn = resolve(doc, sid, h, resolution)
                tid, tn = resolve(doc, sid, t, resolution)
                if hid == tid or (hid, rel, tid, doc, sid) in seen:
                    continue
                seen.add((hid, rel, tid, doc, sid))
                for nid, info in ((hid, hn), (tid, tn)):
                    if nid not in nodes:
                        nodes[nid] = {"id": nid, "node_type": "ENTITY", **info, "entity_class": None}
                props = {"document": doc, "segment": sid, "source": seg["source"],
                         "publication_date": seg["publication_date"],
                         "evidence_text": seg["clean_text"], "trigger": trig,
                         "direction": direction, "confidence": conf,
                         "head_mention": h["text"], "tail_mention": t["text"],
                         "method": "rule_based_v1"}
                records.append({"head": hid, "relation": rel, "tail": tid, **props})
                edges.append({"source": hid, "target": tid, "relation": rel, "properties": props})
                stats[(rel, conf)] += 1

    with open(DATA / "relations.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    base["metadata"].update({
        "name": "CTI Knowledge Graph with rule-based relations", "version": "2.0",
        "relation_schema": {r: {"head": sorted(h), "tail": sorted(t)} for r, (h, t) in SCHEMA.items()},
        "note": "Relations are automatic (trigger phrase + entity types) and not manually validated.",
    })
    graph = {"metadata": base["metadata"], "nodes": list(nodes.values()), "edges": edges}
    with open(DATA / "cti_knowledge_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"Relations extracted: {len(records)}")
    for (rel, conf), n in sorted(stats.items()):
        print(f"  {rel:15s} {conf:7s} {n}")
    print(f"Nodes: {len(nodes)}  Edges: {len(edges)}")


if __name__ == "__main__":
    main()
