# CTI knowledge graph: schema and trust levels

Files (all in `data/`; `cti_knowledge_graph.json` and `relations.jsonl` are the unchanged originals):

| File | Use |
|---|---|
| `cti_knowledge_graph_trusted.json` | Full layered graph. Every edge has a `layer`; every relation has a `trust_level`. |
| `relations_trusted.jsonl` | Only the `trusted` relations, one JSON object per line, ready for retrieval. |
| `relations.jsonl` | All automatic relations (originals, flat). |

## Nodes (`nodes[]`)

| `node_type` | Fields |
|---|---|
| `DOCUMENT` | `id` (`DOC_<report id>`), `document`, `source` (vendor), `publication_date` (`YYYY-MM-DD`) |
| `ENTITY` | `id` (`ENT_<link id>` or `NORM_<TYPE>_<name>`), `name`, `entity_type`, `entity_class`, `link` (Wikipedia, MITRE or NVD URL, or null), `canonical_id`, optional `aliases` |
| `MENTION` | An unresolved mention: `id`, `mention`, `ner_type`, `document`, `segment`, `status` (`UNLINKED` or `AMBIGUOUS`) |

Entity types: `GROUP` (threat actor), `MALWARE`, `TOOL`, `TECHNIQUE`, `TACTIC`, `CVE`, `ORG`, `SECTOR`, `LOC`, `CON`.

## Edges (`edges[]`: `source`, `target`, `relation`, `properties`)

Every edge has `properties.layer`:

| `layer` | `relation` | Meaning |
|---|---|---|
| `mention` | `MENTIONED_IN` | entity or mention -> document (`mention`, `ner_type`, `segment`) |
| `candidate_link` | `CANDIDATE_FOR` | ambiguous mention -> candidate entity (link not resolved) |
| `relation` | `USES`, `USES_TECHNIQUE`, `EXPLOITS`, `TARGETS` | automatic relation, head -> tail |

Relation schema (head -> tail): `USES` group or malware -> malware or tool; `USES_TECHNIQUE` group, malware or tool -> technique or tactic; `EXPLOITS` group, malware or tool -> CVE; `TARGETS` group or malware -> organization, sector or location.

Relation edges carry provenance and trust:

| Property | Meaning |
|---|---|
| `document`, `segment`, `source`, `publication_date` | where the sentence comes from |
| `evidence_text` | the sentence the relation was extracted from |
| `trigger`, `direction` | the phrase that triggered it; `active`, `passive` or `context` |
| `head_mention`, `tail_mention`, `head_type`, `tail_type` | the surface text and NER types of the two mentions |
| `confidence` | `high`, `medium` or `low` (extraction confidence) |
| `trust_level` | `trusted`, `candidate` or `low` (see below) |
| `trust_reasons` | why a relation is not `trusted`, e.g. `malware_or_tool_headed`, `tail_type_mismatch` |
| `human_verified` | always `false`: no relation has been confirmed by a person |
| `method` | `rule_based_v2` |

## Trust levels

| Level | Rule | Use it for |
|---|---|---|
| `trusted` | `USES`/`TARGETS` with a threat-actor head, or `USES_TECHNIQUE`; `high` or `medium` confidence; types match the schema | retrieval and answers, shown with the evidence sentence |
| `candidate` | malware/tool-headed `USES`/`TARGETS`, `EXPLOITS`, a type mismatch, or a duplicate | verification step before it is stated as fact |
| `low` | hedged wording ("likely", "potentially") or a subject borrowed from a nearby sentence | context only |

Basis: a human review of 66 relations from model-tagged unseen reports found actor-headed relations right 75-80% of the time and malware-headed `USES`/`TARGETS` right far less often (see the README). On this gold-tagged graph the `candidate` relations looked about as accurate as the `trusted` ones in a provisional single-reviewer audit, so `candidate` is conservative here. Treat all relations as claims from one sentence of one vendor report, not as facts, and always show `evidence_text`, `source` and `publication_date`. Two reports can disagree.

## Loading

```python
import json
graph = json.load(open("data/cti_knowledge_graph_trusted.json", encoding="utf-8"))
nodes = {n["id"]: n for n in graph["nodes"]}
trusted = [e for e in graph["edges"] if e["properties"].get("trust_level") == "trusted"]
for e in trusted[:3]:
    p = e["properties"]
    print(nodes[e["source"]]["name"], e["relation"], nodes[e["target"]]["name"], "|", p["source"], p["publication_date"])
```

Rebuild with `python src/extract_relations.py` then `python src/build_trusted_graph.py` (the second step validates the graph and stops on any problem).
