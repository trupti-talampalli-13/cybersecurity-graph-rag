# Person 1 report: from CTI reports to a cybersecurity knowledge graph

Scope: data, cyber NLP and the knowledge graph. All figures below come from the files in this repository; the
steps to reproduce them are in the README.

## 1. Summary

| | |
|---|---|
| Corpus | 120 AnnoCTR threat reports (Intel 471, Proofpoint, Zscaler, Lab52, QuoIntelligence), 2016 to 2022 |
| Text | 12,178 sentence-level segments, each keeping its report ID, vendor, publication date and position |
| Entities | 10,689 mentions of 12 types, normalized (2,842 forms) and linked (8,109 linked) |
| Relations | 184 automatic relations (`USES`, `USES_TECHNIQUE`, `EXPLOITS`, `TARGETS`), each with its evidence sentence |
| Graph | 3,812 nodes and 10,960 edges; a layered version labels each relation `trusted`, `candidate` or `low` |
| NER model | RoBERTa-base fine-tuned on AnnoCTR: test entity F1 0.689 |
| Relation accuracy | about 58% correct on unseen text (human review, 66 relations); 75-80% when the subject is a threat actor |

The relations are automatic and **not human-verified**. They should be treated as candidates that carry their
evidence, not as facts.

## 2. Dataset and preprocessing

We use the human-annotated part of AnnoCTR (Lange et al., LREC-COLING 2024; CC BY-SA 4.0): the train, dev and test
splits, 70 + 16 + 34 = 120 reports. The larger `train_ext` split has only general tags (organization, location, date,
sector), so it is used only to test the pipeline on unseen text.

`src/preprocess_reports.py` produces one record per report and one per segment (one empty segment is dropped, leaving
12,178; 698 repeated segments, mostly code snippets, are flagged rather than removed). Each segment keeps the report ID,
vendor, publication date (from the corpus's document creation time, which matches the date in every file name),
segment number, the original `text`, and a `clean_text` with Markdown, URLs and typographic characters normalized.
Entity spans are rebuilt from the gold tags with character offsets into the original text. Reports are
concentrated in 2020 to 2021 (106 of 120); the full range is June 2016 to February 2022.

## 3. Entity extraction and normalization

Entity types come from AnnoCTR's human tags: malware (1,657 mentions), organization (1,578), concept (1,546),
location (1,155), date (1,127), technique (1,110), threat actor `GROUP` (699), tactic (694), sector (552),
code snippet (315) and tool (219). AnnoCTR has no CVE label, so CVE identifiers are added by a regular expression
(37 mentions in 9 reports, 15 distinct CVEs).

Normalization is deliberately conservative: case, spacing and dash variants are merged (2,842 normalized forms, 128 with
more than one surface form), but no semantic merging is attempted. In the graph, spelling variants of one linked entity
(for example TrickBot and Trickbot) are merged.

## 4. Entity linking and ambiguous cases

Mentions are linked to Wikipedia or MITRE ATT&CK entries using AnnoCTR's linking annotations (CVEs link to their NVD
page). Of 10,689 mentions, 8,109 are linked (75.9%); 2,580 are not, and 220 mention-level conflicts have more than one
candidate entity. These become `MENTION` nodes with candidate-link edges instead of being guessed.
Coverage differs by type: malware 1,651 of 1,657, threat actors 688 of 699, tools 217 of 219 and CVEs 37 of 37, but
sectors only 22 of 552, and dates and code snippets are never linked by design. Known problems in the source links:
for example "Ursnif" and "Azerbaijan" are linked to nodes typed `TECHNIQUE`; five relations touch such a node, and the
trusted graph flags them.

## 5. Relation extraction

`src/extract_relations.py` is rule-based. Within a sentence it proposes entity pairs that fit the schema (`USES`: actor
or malware to malware or tool; `USES_TECHNIQUE`: actor, malware or tool to technique or tactic; `EXPLOITS`: to a CVE;
`TARGETS`: actor or malware to organization, sector or location), and keeps a pair only if a trigger phrase joins
them. Passive wording ("distributed by X", "delivered through X") reverses direction.

An audit of the first version showed what the sentence does not support, and the extractor now rejects: comparisons and
overlap ("the same algorithm as", "overlapped with"), reasons ("since", "based on"), impersonation ("posing as"),
negated or disputed claims ("we've never seen TA505 use Emotet"), clauses with a new subject, markdown link titles,
reporting vendors named as victims, organizations that are only part of a product name ("Microsoft Office"), generic
words ("ransomware", "APT") as techniques or actors, self-relations, and a malware named as subject when its group is the
real one. Hedged relations ("likely", "potentially") are kept at `low` confidence. 35 regression tests cover these cases,
28 of which fail on the original extractor. Every relation stores its evidence sentence, trigger, direction, report,
vendor, date, the two mentions and their types, and a confidence.

Result: 184 relation instances (137 distinct triples) from 49 of the 120 reports: `TARGETS` 94, `USES` 70,
`USES_TECHNIQUE` 17, `EXPLOITS` 3. `EXPLOITS` is rare because sentences naming a CVE usually do not name the actor.

## 6. NER evaluation

A RoBERTa-base token classifier was fine-tuned on the AnnoCTR train split (Kaggle GPU, 5 epochs, best epoch chosen on
dev). Scores are entity-level, exact match of span and type.

| Split | Precision | Recall | F1 |
|---|---|---|---|
| Dev | 0.692 | 0.594 | 0.639 |
| Test | 0.724 | 0.657 | 0.689 (macro 0.612) |

Test F1 by type: date 0.87, organization 0.87, location 0.85, malware 0.72, concept 0.68, sector 0.65, tactic 0.59,
threat actor 0.53, technique 0.49, code snippet 0.33, tool 0.15. The cybersecurity-specific types are the weakest and
also the rarest in training. Two other runs did not beat the first: training on train and dev together (test F1 0.675)
and the cybersecurity-pretrained SecureBERT (0.673); each is a single run, so the 1.6-point difference is within noise.

Effect on the graph: running the relation rules on model-tagged text (dev and test) reproduces 58% of the relations found
from gold tags with 64% precision against them (88 reference relations), mainly because 36% of threat actors in the
test set are missed. For the 120 reports the gold-tag graph is therefore the more accurate one.

## 7. Relation quality and error analysis

Four reviews were done; the first three used provisional labels from one reviewer (the assistant that wrote the
extractor), the fourth a human reviewer.

| Review | Data | Result |
|---|---|---|
| 1. First extractor | all 218 relations on gold-tagged reports | 170 true, 29 false, 19 uncertain (85% precision excluding uncertain) |
| 2. Current extractor | 184 relations, same reports | 167 true, 1 false, 16 uncertain; **optimistic**: the rules were written from review 1 |
| 3. Unseen reports, round 1 | 60 `train_ext` reports tagged by the model; 71 relations sampled | 59% true; used to write the final rules |
| 4. Unseen reports, round 2 | 60 different reports; 66 relations | provisional 70% true; **human 58% true** (38 true, 19 false, 9 uncertain; 67% excluding uncertain) |

The human review (round 2) is the headline figure. It was stricter than the provisional labels: the two agree on 77%
of rows (88% for true versus not-true; Cohen's kappa 0.56 over three labels and 0.74 for true versus not-true). By type:
`USES_TECHNIQUE` 79% true, `USES` 52%, `TARGETS` 53%. With 66 relations the estimate is rough (about plus or minus 11
points).

**Who is the subject matters.** In the human review, relations whose head is a threat actor were right 75-80% of the time
(`USES` 8 of 10 true with none false, `TARGETS` 12 of 16), while malware-headed relations were right far less often
(`USES` 3 of 11, `TARGETS` 4 of 14); round 1 showed the same direction. On gold-tagged text the malware-headed relations
looked accurate, so the weakness comes from noisy entities in model-tagged text.

Remaining error types (human review): wrong subject 14, entity error 9 (tagger fragments such as "Kit" or "Evil", a router
vendor typed as a group), wrong object 3, wrong relation type 2. Recall (true relations the extractor misses) has not
been measured.

## 8. The knowledge graph

`data/cti_knowledge_graph_trusted.json` has 3,812 nodes (120 documents, 780 entities, 2,912 unresolved mentions) and 10,960
edges, in three layers: mention to document (10,348 edges, each with vendor and date), candidate links for ambiguous
mentions (428), and relations (184). Every relation carries a `trust_level`:

| Level | Relations | Rule |
|---|---|---|
| `trusted` | 112 | threat-actor-headed `USES`/`TARGETS`, or `USES_TECHNIQUE`, high or medium confidence, types match the schema |
| `candidate` | 61 | malware- or tool-headed `USES`/`TARGETS`, `EXPLOITS`, a type mismatch or a duplicate |
| `low` | 11 | hedged wording or a subject borrowed from a nearby sentence |

`human_verified` is always false, and the build validates endpoints, provenance fields, dates and confidence values.
The policy follows the human review; on these gold-tagged reports the candidate relations looked about as accurate as the
trusted ones, so it is conservative here. The original graph and relation file are unchanged. Field definitions and a
loading example are in `docs/trusted_graph_schema.md`.

## 9. Visualizations

`figures/` holds entity-type and relation counts, an overview of all entity clusters linked by relations, neighborhood
graphs for the best-connected actors and malware (Winnti Group, Leviathan, Cobalt Strike), and
`graph_interactive.html` (hover an edge for its evidence sentence). The figures are deterministic.

## 10. Limitations

- Relations are automatic and not human-verified; unseen-text accuracy is about 58% by a human review of 66 relations,
  and the other labels are provisional single-reviewer judgments.
- Relations exist for only 49 of 120 reports; `EXPLOITS` has 3 instances, so it is too thin to rely on.
- The weakest NER types (tool 0.15, technique 0.49, threat actor 0.53) are the ones the graph depends on most.
- Malware-headed relations from model-tagged text are unreliable; recall was not measured.
- Some linked nodes carry the wrong type in the source linking data; sectors are almost never linked.
- Each relation is one sentence of one vendor report. Reports can disagree, so always show the evidence and date.

## 11. Reproducibility

Steps 1 to 8 use only the Python standard library and take about a minute. A clean copy of the repository reproduced nine
output files, the NER predictions and all figures byte for byte, and all 48 tests pass. The model weights (about 500 MB)
are not stored in Git and must be obtained separately or retrained with `src/train_ner.py`. See the README for the
commands, `requirements.txt` for the optional packages, and `data/*audit*.csv` with
`docs/extracted_relations_audit_instructions.md` for the review sheets.
