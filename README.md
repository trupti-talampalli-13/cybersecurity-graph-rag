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

The repository contains a derived AnnoCTR knowledge-graph artifact:

- `data/knowledge_graph.json`

This graph is an output of the project pipeline, not substitutes for the full original corpus. Intermediate segments, entity tables, candidate sets, predictions, annotations, and audit results are generated or handled separately and are not all tracked in Git; check `.gitignore` and `git status` before adding generated data.

## Knowledge graph pipeline

This part turns 120 AnnoCTR reports (train, dev and test splits) into a knowledge graph of threat actors, malware, tools, techniques and victims. Every relation keeps its evidence sentence, source report and publication date.

### Setup

- Python 3.10. Steps 1-8 below use only the standard library.
- Clone AnnoCTR next to this repository (see Datasets above).
- `pip install -r requirements.txt` is only needed for step 9 and the NER scripts.
- On Windows, keep the repository in a short path. Paths over 260 characters make some corpus files unreadable (`preprocess_reports.py` warns when that happens).

### Run

From the repository root, in order (about a minute in total):

| Step | Command | Output |
|---|---|---|
| 1. Clean and preprocess | `python src/preprocess_reports.py` | `data/reports.jsonl`, `data/processed_segments.jsonl` |
| 2. Entity table | `python src/build_entities.py` | `data/entities.jsonl` |
| 3. Normalize | `python src/normalize_entities.py` | `data/normalized_entities.jsonl` |
| 4. Link | `python src/link_entities.py` | `data/linked_entities_candidates.jsonl` |
| 5. Relation candidates | `python src/build_relation_candidates.py`, then `python src/filter_relation_candidates.py` | `data/annoctr_relation_candidates*.jsonl` |
| 6. Base graph | `python src/build_knowledge_graph.py` | `data/knowledge_graph.json` |
| 7. Relations | `python src/extract_relations.py` | `data/relations.jsonl`, `data/cti_knowledge_graph.json` (intermediate files, not tracked in Git; step 8 reads them) |
| 8. Trusted graph | `python src/build_trusted_graph.py` | `data/cti_knowledge_graph_trusted.json`, `data/relations_trusted.jsonl`, `data/assertions.jsonl`, `data/claims.jsonl` |
| 9. Figures | `python src/visualize_graph.py` | `figures/` |

The steps are deterministic: running them again gives identical files.

### What to use

| File | Contents |
|---|---|
| `data/assertions.jsonl` | One record per assertion: subject, predicate, object, polarity, publication date, scope, evidence sentence, source document, certainty, confidence, trust level, verification status. Use this for retrieval. |
| `data/claims.jsonl` | The assertions grouped by claim (subject, predicate, object): repeats, reports, vendors, dates and whether both polarities occur. The input for comparing assertions. |
| `data/constructed_conflicts.jsonl` | 153 labelled assertion pairs for testing contradiction resolution: supports, contradicts, temporal update, scope variant, complements, insufficient evidence (60 of the 153, so the label mix is not natural), unrelated. 120 include a constructed assertion (marked); 33 use real ones only. Built by `src/build_constructed_conflicts.py` to the written criteria in `src/resolve_assertions.py`, so it checks consistency with those criteria, not that they are right. |
| `data/cti_knowledge_graph_trusted.json` | The graph in layers (mentions, candidate links, relations, negative assertions); relations carry a `trust_level`. |
| `data/relations_trusted.jsonl` | Only the relations marked `trusted`. |
| `figures/` | Count charts, network graphs and `graph_interactive.html`. |

Field definitions and a loading example are in [`docs/trusted_graph_schema.md`](docs/trusted_graph_schema.md).

Trust levels: `trusted` is a threat-actor-headed `USES` or `TARGETS`, or any `USES_TECHNIQUE`, with high or medium confidence. `candidate` needs verification (malware-headed `USES`/`TARGETS`, `EXPLOITS`, type mismatches, any negative assertion). `low` is hedged or borrowed from a nearby sentence. No relation is verified by a person.

### Results

| | |
|---|---|
| Corpus | 120 reports, 12,178 segments |
| Entities | 10,689 mentions, 12 types; 8,109 (76%) linked |
| Relations | 188: `TARGETS` 94, `USES` 70, `USES_TECHNIQUE` 21, `EXPLOITS` 3 |
| Trust levels | 112 trusted, 65 candidate, 11 low |
| NER (RoBERTa-base) | entity F1 0.689 on test, 0.639 on dev |
| Relation accuracy on unseen reports | about 58% correct (human review of 66 relations) |
| Recall | about 10-12% on a 30-sentence check labelled by AI reviewers; 25-29% on the same sentences after the verb-phrase rule (the rule was built from those sentences, and it adds only 4 relations corpus-wide) |

Method, error analysis, the NER model comparison and every audit are in [`docs/person1_report.md`](docs/person1_report.md).

### Known limits

- Relations are automatic and none has been confirmed by a person.
- Relations with a threat-actor subject were right 75-80% of the time on unseen reports; malware-subject `USES`/`TARGETS` far less often.
- Recall is low: the extractor needs a trigger word between two entities. An actor directly followed by a verb-phrase technique ("Leviathan collects passwords") is now handled, but only 4 relations in the corpus have that form; the other misses are of different kinds.
- The 120 reports contain no negative claims, and only 3 of 141 distinct claims occur in more than one report. They cannot test contradiction resolution without added or constructed conflicting statements.
- Polarity, certainty and scope come from simple patterns and have not been evaluated.

### NER model

`src/train_ner.py` fine-tunes a transformer on the AnnoCTR train split and was run on a Kaggle GPU:

```bash
python train_ner.py --model roberta-base --epochs 5 --corpus <folder with train.json, dev.json, test.json> --out <output folder>
```

`python src/predict_ner.py` tags the reports with the trained model, and `python src/compare_gold_vs_predicted.py` shows how model tags change the relations. The weights (`models/ner/ner_model.pth` and `labels.json`, about 500 MB) are not in Git; get them from the project owner or retrain. The RoBERTa tokenizer downloads automatically when there is no local copy.

### Tests and review sheets

```bash
python -m unittest tests.test_extract_relations tests.test_trusted_graph
```

The accuracy reviews are in `data/*audit*.csv` (instructions in `docs/extracted_relations_audit_instructions.md`). The recall check is `data/recall_check_sheet.csv`, scored with `python src/score_recall_check.py <filled sheet>` (instructions in `docs/recall_check_instructions.md`). Labels from AI reviewers are kept separately in `data/llm_review/`.

## Earlier experiments

The AZERG relation-extraction experiments, the relation validators (versions 1 to 6), the NLI and hybrid semantic verifiers, the semantic knowledge graphs and the audit and annotation scripts were removed from the tracked files in a cleanup. They are still in the repository history: restore any of them from the tag `before-cleanup` (for example `git checkout before-cleanup -- src/relation_specific_validator_v6.py`).

## Data Tracking

The full AnnoCTR corpus remains in its upstream repository and is not copied into this repository. The final graph and assertion files under `data/` are tracked. Generated predictions, annotations, audits, and other intermediate outputs are selectively excluded; do not assume that every file under `data/` should be committed. Check redistribution terms before sharing any dataset files.
