# CTI knowledge graph: schema and trust levels

Files (all in `data/`; `cti_knowledge_graph.json` and `relations.jsonl` are the unchanged originals):

| File | Use |
|---|---|
| `cti_knowledge_graph_trusted.json` | Full layered graph. Every edge has a `layer`; every relation has a `trust_level`. |
| `assertions.jsonl` | **Every assertion, positive and negative, with its full context, one JSON object per line. Use this for retrieval and contradiction resolution.** |
| `claims.jsonl` | The same assertions grouped by claim (subject, predicate, object): one record per claim with every assertion about it. The starting point for comparing assertions. |
| `relations_trusted.jsonl` | Only the `trusted` positive assertions, in the same format as `assertions.jsonl`. |
| `relations.jsonl` | All automatic assertions as extracted (originals, flat), including negative ones. |

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
| `relation` | `USES`, `USES_TECHNIQUE`, `EXPLOITS`, `TARGETS` | automatic **positive** assertion, head -> tail |
| `negative_assertion` | the same four names | a sentence saying the relation does **not** hold ("never used", "did not use"). Separate layer on purpose: code that follows `relation` edges cannot mistake it for a positive claim. Always `candidate`. |

Relation schema (head -> tail): `USES` group or malware -> malware or tool; `USES_TECHNIQUE` group, malware or tool -> technique or tactic; `EXPLOITS` group, malware or tool -> CVE; `TARGETS` group or malware -> organization, sector or location.

Relation edges carry provenance and trust:

| Property | Meaning |
|---|---|
| `document`, `segment`, `source`, `publication_date` | where the sentence comes from |
| `evidence_text` | the sentence the relation was extracted from |
| `trigger`, `direction` | the phrase that triggered it; `active`, `passive` or `context` |
| `head_mention`, `tail_mention`, `head_type`, `tail_type` | the surface text and NER types of the two mentions |
| `confidence` | `high`, `medium` or `low` (extraction confidence) |
| `polarity` | `positive`, or `negative` when a negation cue ("never", "not", "n't") directly precedes the trigger |
| `certainty` | `asserted`, or `hedged` ("likely", "potentially", "allegedly"); hedged assertions also have `low` confidence |
| `scope` | best-effort context from the sentence only: `campaign` (a campaign name found by pattern, or null), `campaign_mentioned`, `dates` (dates named in the sentence, as written), `date_mentions` (those that could be parsed into absolute ranges), `unresolved_dates` (those that could not, e.g. a month with no year), `event_time` (earliest start and latest end of the parsed dates, or null), `method`. Not a campaign model |
| `verification_status` | `unverified` until the semantic verifier sets `supported`, `contradicted` or `insufficient_evidence` |
| `trust_level` | `trusted`, `candidate` or `low` (see below) |
| `trust_reasons` | why a relation is not `trusted`, e.g. `malware_or_tool_headed`, `tail_type_mismatch` |
| `human_verified` | always `false`: no relation has been confirmed by a person |
| `method` | `rule_based_v3` |

## Assertion records (`assertions.jsonl`)

One record per assertion, using the field names of the project plan:

| Field | Meaning |
|---|---|
| `assertion_id` | `A00001`, ... (stable for a given build) |
| `claim_key` | `subject_id\|predicate\|object_id`: assertions with the same key make the same claim |
| `claim_text` | the claim as one plain sentence using canonical entity names ("TA505 uses Dridex."), negated for a negative assertion ("TA505 does not use Dridex."); for claim-level verification |
| `claim_text_surface` | the same sentence using the report's own wording for the two entities, usually closer to the evidence sentence |
| `subject`, `subject_id`, `subject_type` | canonical entity name, graph node id and entity type |
| `predicate` | `USES`, `USES_TECHNIQUE`, `EXPLOITS` or `TARGETS` |
| `object`, `object_id`, `object_type` | the same for the object |
| `polarity` | `positive` or `negative` |
| `publication_date` | date of the report (not necessarily when the activity happened) |
| `event_start`, `event_end` | the span of the dates the sentence names, from `scope.event_time` (null when none can be parsed; use `publication_date` as the fallback). A date after "since" is open-ended and runs to the report date. This is the span of dates *mentioned*, which may include dates that are not when the activity happened |
| `scope` | `{campaign, campaign_mentioned, dates, date_mentions, unresolved_dates, event_time, method}` as above. Each `date_mentions` item has `text`, `start`, `end`, `precision` (`day`, `month`, `year`), `kind` (`explicit`, or `relative` for "this year"/"last year", resolved against the report date) and `open_ended` when it follows "since" |
| `evidence_sentence` | the sentence the assertion comes from |
| `source_document`, `source_vendor`, `segment_id` | report id, vendor and sentence position |
| `subject_mention`, `object_mention`, `trigger`, `direction` | surface text and the trigger phrase |
| `certainty`, `confidence` | `asserted`/`hedged`; `high`/`medium`/`low` |
| `trust_level`, `trust_reasons` | `trusted`/`candidate`/`low` and why |
| `verification_status`, `human_verified` | `unverified` and `false` until a verifier or a person changes them |
| `layer`, `method` | `relation` or `negative_assertion`; `rule_based_v3` |

**What the context does and does not give you.** Assertions from different reports can be grouped by (`subject_id`, `predicate`, `object_id`) and compared on `polarity` and `publication_date`. `scope` is only what a single sentence names, so two assertions about different campaigns will often both have `campaign: null`; do not treat a null scope as "same scope". On the 120 gold-tagged reports there are **no** negative assertions and only 3 of 137 distinct claims appear in more than one report, so this corpus does not by itself contain contradictions or temporal updates to resolve.

## Claims (`claims.jsonl`)

One record per distinct claim, collecting every assertion that makes it. Nothing is resolved here; the file only puts the evidence for a claim in one place.

| Field | Meaning |
|---|---|
| `claim_key`, `claim_text`, `subject`, `subject_id`, `predicate`, `object`, `object_id` | the claim, with `claim_text` as its positive plain-sentence form |
| `n_assertions`, `n_positive`, `n_negative`, `polarities` | how many assertions, and which polarities occur |
| `has_polarity_conflict` | true when the claim has both a positive and a negative assertion |
| `n_reports`, `reports`, `n_vendors`, `vendors` | distinct source reports and vendors (several sentences from one report count once) |
| `first_publication_date`, `last_publication_date` | range of the report dates |
| `assertions` | one summary per assertion: `assertion_id`, `polarity`, `publication_date`, `event_start`, `event_end`, `source_document`, `source_vendor`, `certainty`, `trust_level`, `campaign` |

Current build: 141 claims, 3 in more than one report, none with both polarities. A claim repeated in reports from different years (for example APT28 using Zebrocy, in 2020 and 2021) is repeated support over time, not a conflict. `has_polarity_conflict` is only a candidate flag: whether two assertions really conflict also depends on scope and time, which this file does not decide.

## Trust levels

| Level | Rule | Use it for |
|---|---|---|
| `trusted` | `USES`/`TARGETS` with a threat-actor head, or `USES_TECHNIQUE`; `high` or `medium` confidence; types match the schema; positive polarity | retrieval and answers, shown with the evidence sentence |
| `candidate` | malware/tool-headed `USES`/`TARGETS`, `EXPLOITS`, a type mismatch, a duplicate, or any negative assertion | verification step before it is stated as fact |
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
