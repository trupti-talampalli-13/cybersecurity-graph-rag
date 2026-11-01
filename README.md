# cybersecurity-graph-rag
Hybrid and Graph-Augmented RAG for Cyber Threat Intelligence

## Datasets

### AnnoCTR: Primary Corpus

The CTI pipeline is built around AnnoCTR. Its annotated reports provide the text segments and entity labels used for entity extraction, normalization, linking, relation-candidate generation, and knowledge-graph construction.

The full AnnoCTR corpus is **not bundled in this repository**. It is maintained upstream at [boschresearch/anno-ctr-lrec-coling-2024](https://github.com/boschresearch/anno-ctr-lrec-coling-2024). The source README states that the corpus is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Follow that license's attribution and ShareAlike terms, and cite the dataset paper when using it:

> Lukas Lange, Marc Müller, Ghazaleh Haratinezhad Torbati, Dragan Milchevski, Patrick Grau, Subhash Pujari, and Annemarie Friedrich. “AnnoCTR: A Dataset for Detecting and Linking Entities, Tactics, and Techniques in Cyber Threat Reports.” LREC-COLING 2024. [arXiv:2404.07765](https://arxiv.org/abs/2404.07765).

The upstream checkout is about 558 MB in the local copy used during development, so this project references it rather than duplicating the corpus. To use the current AnnoCTR preparation script, clone it beside this repository so the paths look like:

```text
workspace/
	anno-ctr-lrec-coling-2024/
	cti_graph_rag/
```

From the `cti_graph_rag` directory:

```powershell
git clone https://github.com/boschresearch/anno-ctr-lrec-coling-2024.git ../anno-ctr-lrec-coling-2024
python src/prepare_annocr.py
```

`src/prepare_annocr.py` reads `../anno-ctr-lrec-coling-2024/AnnoCTR/ner_json/train.json` only. It uses the provided `all_tags` BIO labels to reconstruct entity spans, takes document and segment IDs from each record ID, and writes JSON Lines records with `document`, `segment`, `text`, and `entities` to `data/train_segments.jsonl`. It does not preprocess the original corpus in place, and this step does not combine the train, development, and test splits.

The repository contains derived AnnoCTR knowledge-graph artifacts:

- `data/knowledge_graph.json`
- `data/semantic_knowledge_graph.json`
- `data/semantic_knowledge_graph_verified.json`

These graphs are outputs of the project pipeline, not substitutes for the full original corpus. Intermediate segments, entity tables, candidate sets, predictions, annotations, and audit results are generated or handled separately and are not all tracked in Git; check `.gitignore` and `git status` before adding generated data.

### AZERG: Auxiliary Relation-Extraction Dataset

AZERG is a separate benchmark used for relation-extraction experiments; it is not the source corpus for the AnnoCTR knowledge graph. The dataset is published as [QCRI/AZERG-Dataset on Hugging Face](https://huggingface.co/datasets/QCRI/AZERG-Dataset). `data/azerg/` in this repository contains T3 and T4 train/test files, plus cleaned T4 train/test JSONL files.

To download the four source splits and regenerate the cleaned T4 files, run from the repository root:

```powershell
python src/download_azerg.py
python src/prepare_azerg.py
```

`src/prepare_azerg.py` extracts source entity, target entity, passage text, and the labeled relation from the T4 records. It writes `data/azerg/t4_train_clean.jsonl` and `data/azerg/t4_test_clean.jsonl`. The T3 files are retained in their original format by the download step.

## CTI Knowledge Graph Pipeline (Data, NER and Graph)

This pipeline turns the AnnoCTR reports (train, dev and test: 120 reports, 12,178 segments after dropping one empty segment) into a cybersecurity knowledge graph. Run each step from the repository root, in order:

| Step | Script | Output |
|---|---|---|
| 1. Clean and preprocess | `python src/preprocess_reports.py` | `data/reports.jsonl`, `data/processed_segments.jsonl` |
| 2. Entity table | `python src/build_entities.py` | `data/entities.jsonl` |
| 3. Normalize | `python src/normalize_entities.py` | `data/normalized_entities.jsonl` |
| 4. Link | `python src/link_entities.py` | `data/linked_entities_candidates.jsonl` |
| 5. Relation candidates | `python src/build_relation_candidates.py` then `python src/filter_relation_candidates.py` | `data/annoctr_relation_candidates*.jsonl` |
| 6. Base graph | `python src/build_knowledge_graph.py` | `data/knowledge_graph.json` |
| 7. Relations and final graph | `python src/extract_relations.py` | `data/relations.jsonl`, `data/cti_knowledge_graph.json` |
| 8. Trusted graph | `python src/build_trusted_graph.py` | `data/cti_knowledge_graph_trusted.json`, `data/relations_trusted.jsonl` |
| 9. Visualizations (needs `networkx`, `matplotlib`) | `python src/visualize_graph.py` | `figures/` |

**Setup.** Steps 1-8 use only the Python standard library (Python 3.10 was used) and take about a minute in total. Clone the AnnoCTR corpus beside this repository as described above, then run the steps from the repository root. The model steps (`train_ner.py`, `predict_ner.py`, `unseen_relation_check.py`) and step 9 need the optional packages in `requirements.txt` (`pip install -r requirements.txt`). The trained weights (`models/ner/ner_model.pth`, `models/ner/labels.json`, about 500 MB) are **not stored in Git**: get them from the project owner or retrain with `train_ner.py` (see "Trained NER Model"); the RoBERTa tokenizer is downloaded automatically from the Hugging Face Hub when no local copy is present. A clean copy of the repository was checked: steps 1-8 reproduce `reports.jsonl`, `processed_segments.jsonl`, `entities.jsonl`, `linked_entities_candidates.jsonl`, `knowledge_graph.json`, `relations.jsonl`, `cti_knowledge_graph.json`, `cti_knowledge_graph_trusted.json` and `relations_trusted.jsonl` byte for byte. Run the tests with `python -m unittest tests.test_extract_relations tests.test_trusted_graph`. On Windows, keep the repository in a short path: paths over 260 characters make some corpus files unreadable (the preprocessing script now warns when report titles cannot be read).

**Preprocessing.** Every segment keeps its report ID, source vendor, publication date (from the AnnoCTR document creation time) and segment number. `text` is the original sentence and `clean_text` has Markdown, URLs and curly quotes normalized. Entity spans are rebuilt from the gold BIO tags with character offsets into `text`. `data/processed_segments.jsonl` contains the corpus text, so it is generated locally and not tracked.

**Entities.** Types come from the AnnoCTR gold tags (`GROUP` is the threat-actor type, plus `MALWARE`, `TOOL`, `TECHNIQUE`, `TACTIC`, `ORG`, `SECTOR`, `LOC`, `CON`, `DATE`). AnnoCTR has no CVE label, so `CVE-YYYY-NNNN` ids are added by regex. Mentions are normalized (case, spacing, dashes) and linked to Wikipedia/MITRE entries using the AnnoCTR linking annotations. CVEs link to their NVD page. Mentions without a link stay as `MENTION` nodes.

**Relations.** `extract_relations.py` proposes entity pairs within a segment, keeps those whose types fit the schema and that are joined by a trigger phrase, and handles passive wording ("distributed by X"):

| Relation | Head | Tail |
|---|---|---|
| `USES` | group, malware | malware, tool |
| `USES_TECHNIQUE` | group, malware, tool | technique, tactic |
| `EXPLOITS` | group, malware, tool | CVE |
| `TARGETS` | group, malware | organization, sector, location |

Each relation edge stores its evidence sentence, trigger, report, date and a `high`/`medium`/`low` confidence. Entity nodes that are spelling variants of one entity (for example TrickBot and Trickbot) are merged and listed in `aliases`. Relations are automatic and not human-validated; see "Relation audit" below. `EXPLOITS` finds only a few cases because CVE sentences often do not name the actor.

The extractor (`rule_based_v2`) refuses candidates the sentence does not assert: comparisons and overlap ("the same algorithm as", "overlapped with"), reasons ("since", "based on"), impersonation ("posing as"), clauses with a new subject ("they have deployed"), markdown link titles, vendors or sources of reporting named as victims, generic words such as "ransomware" as techniques, and a malware named as subject when its group is the real one. Passive phrasing with "by", "through" or "via" reverses direction, and hedged relations ("likely", "potentially") are kept at `low` confidence. Regression tests for each of these are in `tests/test_extract_relations.py` (`python -m unittest tests.test_extract_relations`).

**Relation audit.** `data/extracted_relations_audit_provisional.csv` labels all 218 relations of the first extractor version (`data/relations_rule_based_v1.jsonl`) as TRUE / FALSE / UNCERTAIN against their evidence sentence, using `docs/extracted_relations_audit_instructions.md`. **These labels are provisional: one reviewer (the assistant that wrote the extractor), not human-verified.** Result for v1: 170 TRUE, 29 FALSE, 19 UNCERTAIN (85% precision excluding uncertain; 90% for `TARGETS`, 82% for `USES`, 74% for `USES_TECHNIQUE`). The v2 rules were written from those errors and remove 28 of the 29 FALSE rows while losing 4 of 170 TRUE ones, leaving 184 relations (`data/extracted_relations_audit_provisional_v2.csv`, 167 TRUE, 1 FALSE, 16 UNCERTAIN). That v2 figure is in-sample and therefore optimistic; recall (relations the extractor misses) has not been measured. The audit CSVs and summary (`data/extracted_relations_audit_summary.json`) are the inputs for a human review.

**Unseen-report check (end to end).** To measure accuracy without that bias, `src/unseen_relation_check.py` tagged 60 randomly chosen reports from AnnoCTR `train_ext` (outside the 120 gold reports) with the trained NER model and ran the v2 rules (122 relations found; 71 sampled, stratified by type; same provisional single-reviewer labels, `data/unseen_check_audit_provisional.csv`). Result: 42 TRUE, 25 FALSE, 4 UNCERTAIN, i.e. about 59% TRUE (63% excluding uncertain), much lower than the in-sample figure. By type: `USES_TECHNIQUE` 82% TRUE, `TARGETS` 63%, `USES` 44%. Errors are mostly from the tagger and from context the rules do not read: 11 entity errors (for example "Microsoft" inside "Microsoft Office" read as a victim, a campaign name or an author's handle read as an actor or malware), 4 negated or disputed claims ("we've never seen TA505 use Emotet"), and wrong subject, direction or relation type. High-confidence relations were right 75% of the time against 54% for medium. Treat the automatic relations as candidates for verification, not as facts.

Rules were then added for these failures (negated or disputed claims, organizations that are part of a product name such as "Microsoft Office", malware-description words such as "Banking Trojan", self-relations, generic actor words, "used in", "reminiscent of", research fields). Because those rules were written from the round 1 errors, a second round (`python src/unseen_relation_check.py 2`) tagged 60 different unseen reports that were not used in round 1 or when writing the rules (78 relations found, 66 reviewed, `data/unseen_check_audit_provisional_round2.csv`). Result: 46 TRUE, 16 FALSE, 4 UNCERTAIN, i.e. about 70% TRUE (74% excluding uncertain). By type: `TARGETS` 80%, `USES_TECHNIQUE` 79%, `USES` 52%, `EXPLOITS` 0 of 1; 84% precision for high confidence and 71% for medium. The remaining errors are mostly from the tagger (8 of 16: fragments such as "Kit" or "Evil", a router vendor or ISP typed as a group) and relation-type confusions (6: comparisons, "same operator deploys both", lists of malware categories). With 66 reviewed relations the estimate is rough (about ±11 points), the two rounds cover different reports and different mixes of relation types, and all labels are provisional single-reviewer judgments, so they are not a clean before/after comparison.

**Human review of round 2.** A project member then labelled the same 66 relations (`data/unseen_check_audit_human_round2.csv`) from the evidence sentences. Human result: 38 TRUE, 19 FALSE, 9 UNCERTAIN, i.e. about 58% TRUE (67% excluding uncertain); by type `USES_TECHNIQUE` 79%, `USES` 52%, `TARGETS` 53%. The human review was stricter than the provisional labels: the two agree on 77% of rows (88% on TRUE versus not-TRUE; Cohen's kappa 0.56 over three labels, 0.74 for TRUE versus not-TRUE). The disagreements are mostly relations where a malware family is the subject of "used to target" (the human marked them UNCERTAIN or FALSE because the actor is not named) and entity fragments from the tagger (human FALSE, provisional UNCERTAIN). **Treat the human-reviewed figure as the headline number for relations extracted from model-tagged text.** The human error categories were wrong subject 14, entity error 9, wrong object 3, wrong relation type 2.

**Who is the subject matters.** In the human-reviewed round, relations whose head is a threat actor were right about 75-80% of the time (`USES` 8 of 10 TRUE, none FALSE; `TARGETS` 12 of 16), while relations whose head is a malware family were right far less often (`USES` 3 of 11, `TARGETS` 4 of 14). The same direction appeared in round 1 (actor-headed `TARGETS` 75% against 40% for malware-headed). On the 120 gold-tagged reports the malware-headed relations were accurate (88-95% under the provisional labels), so the weakness comes from noisy entities in model-tagged text, and downstream users should prefer actor-headed relations and treat malware-headed `USES`/`TARGETS` from model-tagged text as low trust.

**Known issues.** Five of the 184 relation instances touch a node whose linked type disagrees with the mention type (for example "Ursnif" and "Azerbaijan" are linked to nodes typed `TECHNIQUE` in the AnnoCTR linking annotations), and one relation is counted twice because two aliases of the same actor (APT-C-36 and Blind Eagle) were merged. Relation precision has only been checked by a provisional single-reviewer audit (see above).

**Visualizations.** `figures/` holds entity and relation count charts, an overview graph, neighborhood graphs for the best-connected actors, and `graph_interactive.html` (hover an edge for its evidence; loads vis-network from a CDN).

## Trusted Graph (layers and trust levels)

`python src/build_trusted_graph.py` writes `data/cti_knowledge_graph_trusted.json`, a layered copy of the graph in which the original graph and `relations.jsonl` stay unchanged. Every edge has a layer (mention, candidate link, relation) and every automatic relation has a `trust_level`: `trusted` (112 relations: threat-actor-headed `USES`/`TARGETS`, or `USES_TECHNIQUE`, with high or medium confidence), `candidate` (61: malware- or tool-headed `USES`/`TARGETS`, `EXPLOITS`, type mismatches, duplicates) and `low` (11: hedged or borrowed subject). `human_verified` is always false. The trusted relations alone are in `data/relations_trusted.jsonl`. The graph is validated on every build (endpoints, provenance fields, ISO dates, confidence and trust values), and the policy is tested in `tests/test_trusted_graph.py`. The policy follows the human review described below; on these gold-tagged reports the candidate relations looked about as accurate as the trusted ones in the provisional audit, so it is conservative. Field definitions and a loading example are in `docs/trusted_graph_schema.md`.

## Trained NER Model

Besides using the AnnoCTR gold tags, the project includes a fine-tuned NER model so new reports can be tagged automatically.

| Script | Purpose |
|---|---|
| `src/train_ner.py` | Fine-tunes a transformer for token classification on AnnoCTR train, selects the best epoch on dev, and scores dev and test. Uses a GPU when available. |
| `src/predict_ner.py` | Loads `models/ner/ner_model.pth` and tags every sentence of the three splits (`data/predicted_entities.jsonl`, `data/ner_vs_gold.json`). |
| `src/compare_gold_vs_predicted.py` | Runs the relation extractor on gold and on predicted entities and compares them (`data/ner_graph_comparison.json`). |

The model was trained on a Kaggle GPU:

```bash
python train_ner.py --model roberta-base --epochs 5 --corpus <folder with train.json, dev.json, test.json> --out <output folder>
```

It writes `ner_model/ner_model.pth` (PyTorch weights), `ner_model/labels.json`, `ner_eval.json` and `ner_predictions_test.jsonl`. The weights (about 500 MB) are not tracked in Git.

**Results** (roberta-base, 5 epochs, best epoch 3 by dev; entity-level exact match of span and type):

| Split | Precision | Recall | F1 |
|---|---|---|---|
| Dev | 0.692 | 0.594 | 0.639 |
| Test | 0.724 | 0.657 | 0.689 |

Test F1 by type: DATE 0.87, ORG 0.87, LOC 0.85, MALWARE 0.72, CON 0.68, SECTOR 0.66, TACTIC 0.59, GROUP 0.53, TECHNIQUE 0.49, CLICommand/CodeSnippet 0.33, TOOL 0.15. The cybersecurity-specific types (threat actors, tools, techniques) are the weakest; they are also the rarest in training.

**Model comparison.** Two further runs were tried on Kaggle (scores copied from the training logs; each is a single run with one random seed, test set never used for training or model selection):

| Model | Precision | Recall | F1 | Macro-F1 |
|---|---|---|---|---|
| roberta-base, train only (used) | 0.724 | 0.657 | 0.689 | 0.612 |
| roberta-base, train + dev, 3 epochs | 0.681 | 0.668 | 0.675 | 0.592 |
| SecureBERT (`ehsanaghaei/SecureBERT`), train only, 5 epochs | 0.690 | 0.657 | 0.673 | 0.591 |

| Test F1 | GROUP | MALWARE | TACTIC | TECHNIQUE | TOOL |
|---|---|---|---|---|---|
| roberta-base, train only | 0.533 | 0.716 | 0.590 | 0.488 | 0.146 |
| roberta-base, train + dev | 0.471 | 0.716 | 0.625 | 0.394 | 0.052 |
| SecureBERT | 0.586 | 0.718 | 0.618 | 0.399 | 0.077 |

Adding dev to training did not help (its dev score is not held-out and is not reported). SecureBERT improved `GROUP` but lost on `TECHNIQUE` and `TOOL`. The differences are within about 1.6 F1 points on single runs, so none of the variants is clearly better, and the original model is kept. SecureBERT is released under the BigScience OpenRAIL-M licence.

**Effect on the graph.** On dev and test text, running the relation rules on model-tagged entities reproduces about 58% of the relations found from gold tags, with about 64% precision against them. Missed `GROUP` entities (136 predicted against 213 gold in test) are the main cause. These counts are small (88 reference relations) and the reference relations are themselves automatic, so treat them as indicative. For the 120 AnnoCTR reports the gold-tag graph remains the more accurate one.

## Semantic Relation Verification

The V1 NLI and Hybrid outputs are preserved as baselines. The separate,
deterministic V2 rule verifier and its conservative decision policy are
documented in [`docs/semantic_verifier_v2_implementation.md`](docs/semantic_verifier_v2_implementation.md).

## Data Tracking

The full AnnoCTR corpus remains in its upstream repository and is not copied into this repository. Selected AZERG input splits and the three derived knowledge-graph JSON files are tracked here. Generated predictions, annotations, audits, and other intermediate outputs are selectively excluded; do not assume that every file under `data/` should be committed. Check redistribution terms before sharing any dataset files.
