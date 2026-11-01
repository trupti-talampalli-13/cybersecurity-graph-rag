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

NOT_PASSIVE = r"(?!\s+(?:by|through|via|from|as)\b)"   # "delivered by/through X", "used as a payload" are passive
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
    "USES": r"\b(us|utili[sz]|employ|deploy|drop|deliver|distribut|install|download|load|launch|execut)ed"
            r"\s+(?:by|through|via)\b|\b(?:us|utili[sz])ed\s+in\b",   # "Crypter used in Emotet": Emotet uses it
    "USES_TECHNIQUE": r"\b(us|employ|perform|leverag|conduct|implement|utili[sz])ed\s+by\b",
    "EXPLOITS": r"\b(exploit|leverag|abus|weaponi[sz])ed\s+by\b",
    "TARGETS": r"\b(target|attack|hit|victimi[sz]|compromis|infect|affect)ed\s+by\b",
}
ACTIVE_RE = {r: re.compile(p, re.I) for r, p in ACTIVE.items()}
PASSIVE_RE = {r: re.compile(p, re.I) for r, p in PASSIVE.items()}
# max characters between the two mentions, per relation (and for passive phrasing)
MAX_GAP = {"USES": 120, "USES_TECHNIQUE": 70, "EXPLOITS": 160, "TARGETS": 100}
MAX_PASSIVE_GAP = 60
MAX_PASSIVE_AGENT_GAP = 30   # characters allowed between "by" and the agent that follows it
MAX_PREP_GAP = 20            # characters allowed between "via"/"through" and its object
MAX_TAIL_WORDS = {"USES_TECHNIQUE": 4}
STRONG_TARGET = re.compile(r"target\w*|aim\w*|against|focus\w* on|go(?:es|ing)? after|victimi[sz]\w*", re.I)

# Text between head and tail that means the relation is not asserted between these two entities:
# comparisons, shared infrastructure/overlap, reasons, relative clauses with a new subject, "other X such as Y"
BLOCK_BETWEEN = re.compile(
    r"\bsame\b[^.,;]{0,40}\b(?:as|that|manner)\b|\b(?:similar|overlap\w*|compar\w*|alike)\b|\bin line with\b"
    r"|\b(?:since|because|due to|based on)\b|\bother\b[^.]*\bsuch as\b|\bthat\s+(?:could|would|may|might|can)\b"
    r"|\b(?:reminiscent|resembl\w+|background in|specializ\w+ in)\b|\b(?:connected|tied|linked|related) to\b",
    re.I)
# Negated or disputed claims in the words around the pair ("we've never seen TA505 use Emotet")
NEGATION = re.compile(r"\b(?:never|not|cannot|neither|nor|incorrectly|wrongly|falsely|disputed|unexplained)\b|n't\b",
                      re.I)
PRONOUN_SUBJECT = re.compile(r"\b(?:they|he|she)\s+(?:\w+\s+){0,2}$", re.I)   # clause has a new (human) subject
IMPERSONATION = re.compile(r"\b(?:posing as|impersonat\w+|masquerad\w+|claim(?:s|ed|ing)? to)\b", re.I)
HEDGE = re.compile(r"\b(?:likely|probably|possibly|potentially|potential|allegedly|presumably|might|could|would|"
                   r"suspect\w*|appear(?:s|ed)? to|seem(?:s|ed)? to|assess\w* that)\b", re.I)
HEAD_ROLE_NOUN = re.compile(r"^\s+(?:operators?|developers?|authors?|actors?|gang|affiliates?|creators?)\b", re.I)
REPORTER_AFTER = re.compile(r"^\s+\[?(?:reporting|reports?|research|researchers?|analysts?|assess\w*)\b", re.I)
REPORTER_BEFORE = re.compile(r"(?:expert|analyst|researcher|spokesperson)\s+(?:in|at|from|with)\s+(?:the\s+)?$", re.I)
REPORTER_NAMES = {"proofpoint", "zscaler", "intel 471", "intel471", "lab52", "quointelligence", "threatlabz",
                  "mandiant", "eset", "kaspersky", "fireeye", "crowdstrike", "symantec", "trend micro"}
GENERIC_TECHNIQUE_TAILS = {"ransomware", "malware", "trojan", "backdoor", "botnet"}
GENERIC_HEADS = {"apt", "actor", "actors", "group", "threat actor", "attacker", "attackers"}
# "Microsoft Office", "Oracle WebLogic": the organization is part of a product name, not a victim
PRODUCT_AFTER = re.compile(r"^\s+(?:Office|Word|Excel|PowerPoint|Outlook|OneNote|Windows|Exchange|Defender|Azure|"
                           r"SharePoint|Teams|WebLogic|SQL|Server|Edge|Explorer|Flash|Java|Acrobat|Reader)\b", re.I)
# "Banking Trojan", "initial access malware": the word belongs to a malware description
MALWARE_CLASS_AFTER = re.compile(r"^\s+(?:trojans?|malware|ransomware|botnet|backdoor|stealer|loader|rat|payloads?)\b",
                                 re.I)
LINK_TARGET = re.compile(r"\]\([^)]*\)")   # markdown link target/title: not prose


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


def mask_links(text):
    """Blank out markdown link targets/titles (same length, so offsets stay valid)."""
    chars = list(text)
    spans = []
    for m in LINK_TARGET.finditer(text):
        spans.append((m.start(), m.end()))
        for i in range(m.start(), m.end()):
            chars[i] = " "
    return "".join(chars), spans


def candidate_ok(rel, h, t, trig, direction, text, between, ents):
    """Reject a (head, trigger, tail) candidate that the sentence does not really assert."""
    if BLOCK_BETWEEN.search(between):
        return False
    if norm(h["text"]) == norm(t["text"]) or norm(h["text"]) in GENERIC_HEADS:
        return False           # an entity cannot target itself; "APT"/"group" alone is not an actor name
    lo = min(h["char_start"], t["char_start"])
    hi = max(h["char_end"], t["char_end"])
    if IMPERSONATION.search(text[max(0, lo - 80):hi]) or NEGATION.search(text[max(0, lo - 60):hi]):
        return False
    if direction == "active":
        if PRONOUN_SUBJECT.search(between[:trig.start()]):
            return False
        if rel in ("USES", "USES_TECHNIQUE", "EXPLOITS") and h["type"] in ("MALWARE", "TOOL") \
                and HEAD_ROLE_NOUN.match(text[h["char_end"]:]):
            return False       # "<malware> operators/actors ..." is about the people, not the malware
        if trig.group(0).lower() in ("via", "through") and len(between) - trig.end() > MAX_PREP_GAP:
            return False
        if trig.group(0).lower() in ("used", "utilized", "utilised") and re.match(r"\s+in\b", between[trig.end():]):
            return False       # "X used in Y" means Y uses X (handled as passive)
    else:
        if len(between) - trig.end() > MAX_PASSIVE_AGENT_GAP:
            return False       # the agent must follow "by" directly
        agent_from = t["char_end"] + trig.end()
        if any(e is not h and e is not t and agent_from <= e["char_start"] and e["char_end"] <= h["char_start"]
               for e in ents):
            return False       # another entity sits between "by" and the head, so it is not the agent
    if rel == "USES_TECHNIQUE" and norm(t["text"]) in GENERIC_TECHNIQUE_TAILS:
        return False           # "ransomware" etc. name a malware class, not a technique
    if rel == "USES_TECHNIQUE" and MALWARE_CLASS_AFTER.match(text[t["char_end"]:]):
        return False           # "initial access malware": describes the malware, not a technique it uses
    if rel == "TARGETS":
        if norm(t["text"]) in REPORTER_NAMES:
            return False       # the vendor reporting on the activity is not the victim
        if PRODUCT_AFTER.match(text[t["char_end"]:]) or MALWARE_CLASS_AFTER.match(text[t["char_end"]:]):
            return False       # "Microsoft Office", "Banking Trojan": part of a product/malware name
        if REPORTER_AFTER.match(text[t["char_end"]:]) or REPORTER_BEFORE.search(text[:t["char_start"]]):
            return False
        if t["type"] == "LOC" and re.match(r"\s+(?:Sea|Ocean)\b", text[t["char_end"]:]):
            return False       # "Chinese Sea" is a place name, not a target
    return True


def find_relations(seg, resolution):
    text, spans = mask_links(seg["text"])
    ents = [e for e in seg["entities"] if e["char_start"] is not None
            and not any(a <= e["char_start"] and e["char_end"] <= b for a, b in spans)]
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
                if not candidate_ok(rel, h, t, trig, direction, text, between, ents):
                    continue
                lo, hi = min(h["char_start"], t["char_start"]), max(h["char_end"], t["char_end"])
                inner = [e for e in ents if e not in (h, t)
                         and min(h["char_end"], t["char_end"]) <= e["char_start"]
                         and e["char_end"] <= max(h["char_start"], t["char_start"])
                         and (e["type"] in head_types or e["type"] in tail_types)]
                if HEDGE.search(text[max(0, lo - 60):hi]):
                    conf = "low"       # hedged ("likely", "potentially"...): kept but not trusted
                else:
                    conf = "high" if not inner and len(between) <= 40 else "medium"
                out.append((rel, h, t, trig.group(0).lower(), direction, conf))
    # "TA415 used Cobalt Strike in a campaign against X": the group, not its tool, is the subject
    uses = {(id(h), id(t)) for rel, h, t, *_ in out if rel == "USES" and h["type"] == "GROUP"}
    group_targets = {(id(h), id(t)) for rel, h, t, *_ in out if rel == "TARGETS" and h["type"] == "GROUP"}
    used_by_group = {used_id for _, used_id in uses}
    keep = []
    for item in out:
        rel, h, t, direction = item[0], item[1], item[2], item[4]
        if rel == "TARGETS" and h["type"] in ("MALWARE", "TOOL") and any(
                (gid, id(h)) in uses and (gid, id(t)) in group_targets for gid, _ in uses):
            continue
        # "<group> ... using <tool>, targeting/using X": the clause after the comma still has the group as subject
        if rel in ("USES_TECHNIQUE", "TARGETS") and direction == "active" and id(h) in used_by_group \
                and "," in text[h["char_end"]:t["char_start"]]:
            continue
        keep.append(item)
    return keep


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


GENERIC_SUFFIX = re.compile(r"(group|apt|team|ransomware|malware|trojan|loader|botnet|rat|stealer|"
                            r"banking|backdoor)")
MERGE_TYPES = {"GROUP", "MALWARE", "TOOL", "TECHNIQUE", "TACTIC", "CVE"}


def alias_key(node):
    """Same type + same name once case, punctuation and generic words (group, trojan...) are dropped."""
    name = (node.get("name") or "").lower()
    if node["entity_type"] in ("GROUP", "MALWARE", "TOOL"):
        name = GENERIC_SUFFIX.sub("", name) or name
    return node["entity_type"], re.sub(r"[^a-z0-9]", "", name)


def merge_aliases(nodes, edges, records):
    """Collapse entity nodes that are spelling/linking variants of one entity.
    Prefers a linked (ENT_) node as the survivor and records the others in `aliases`."""
    groups = {}
    for n in nodes.values():
        if n["node_type"] == "ENTITY" and n.get("entity_type") in MERGE_TYPES and n.get("name"):
            groups.setdefault(alias_key(n), []).append(n)
    remap = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda n: (n["id"].startswith("NORM_"), n["id"]))
        keep = members[0]
        keep["aliases"] = sorted({m["name"] for m in members[1:]} - {keep["name"]})
        for m in members[1:]:
            remap[m["id"]] = keep["id"]
            del nodes[m["id"]]
    for e in edges:
        e["source"], e["target"] = remap.get(e["source"], e["source"]), remap.get(e["target"], e["target"])
    for r in records:
        r["head"], r["tail"] = remap.get(r["head"], r["head"]), remap.get(r["tail"], r["tail"])
    return remap


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
                         "head_type": h["type"], "tail_type": t["type"],   # NER types of the two mentions
                         "method": "rule_based_v2"}
                records.append({"head": hid, "relation": rel, "tail": tid, **props})
                edges.append({"source": hid, "target": tid, "relation": rel, "properties": props})
                stats[(rel, conf)] += 1

    remap = merge_aliases(nodes, edges, records)
    print(f"Merged {len(remap)} alias nodes into their canonical entities")

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
