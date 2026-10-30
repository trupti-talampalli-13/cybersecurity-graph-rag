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

This pipeline turns the AnnoCTR reports (train, dev and test: 120 reports, 12,179 segments) into a cybersecurity knowledge graph. Run each step from the repository root, in order:

| Step | Script | Output |
|---|---|---|
| 1. Clean and preprocess | `python src/preprocess_reports.py` | `data/reports.jsonl`, `data/processed_segments.jsonl` |
| 2. Entity table | `python src/build_entities.py` | `data/entities.jsonl` |
| 3. Normalize | `python src/normalize_entities.py` | `data/normalized_entities.jsonl` |
| 4. Link | `python src/link_entities.py` | `data/linked_entities_candidates.jsonl` |
| 5. Relation candidates | `python src/build_relation_candidates.py` then `python src/filter_relation_candidates.py` | `data/annoctr_relation_candidates*.jsonl` |
| 6. Base graph | `python src/build_knowledge_graph.py` | `data/knowledge_graph.json` |
| 7. Relations and final graph | `python src/extract_relations.py` | `data/relations.jsonl`, `data/cti_knowledge_graph.json` |
| 8. Visualizations | `python src/visualize_graph.py` | `figures/` |

**Preprocessing.** Every segment keeps its report ID, source vendor, publication date (from the AnnoCTR document creation time) and segment number. `text` is the original sentence and `clean_text` has Markdown, URLs and curly quotes normalized. Entity spans are rebuilt from the gold BIO tags with character offsets into `text`. `data/processed_segments.jsonl` contains the corpus text, so it is generated locally and not tracked.

**Entities.** Types come from the AnnoCTR gold tags (`GROUP` is the threat-actor type, plus `MALWARE`, `TOOL`, `TECHNIQUE`, `TACTIC`, `ORG`, `SECTOR`, `LOC`, `CON`, `DATE`). AnnoCTR has no CVE label, so `CVE-YYYY-NNNN` ids are added by regex. Mentions are normalized (case, spacing, dashes) and linked to Wikipedia/MITRE entries using the AnnoCTR linking annotations. CVEs link to their NVD page. Mentions without a link stay as `MENTION` nodes.

**Relations.** `extract_relations.py` proposes entity pairs within a segment, keeps those whose types fit the schema and that are joined by a trigger phrase, and handles passive wording ("distributed by X"):

| Relation | Head | Tail |
|---|---|---|
| `USES` | group, malware | malware, tool |
| `USES_TECHNIQUE` | group, malware, tool | technique, tactic |
| `EXPLOITS` | group, malware, tool | CVE |
| `TARGETS` | group, malware | organization, sector, location |

Each relation edge stores its evidence sentence, trigger, report, date and a `high`/`medium`/`low` confidence. Entity nodes that are spelling variants of one entity (for example TrickBot and Trickbot) are merged and listed in `aliases`. Relations are automatic and not manually validated; in a hand check of a small sample `TARGETS` and `USES` were mostly correct, `USES_TECHNIQUE` was noisy, and `EXPLOITS` finds only a few cases because CVE sentences often do not name the actor.

**Visualizations.** `figures/` holds entity and relation count charts, an overview graph, neighborhood graphs for the best-connected actors, and `graph_interactive.html` (hover an edge for its evidence; loads vis-network from a CDN).

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

**Effect on the graph.** On dev and test text, running the relation rules on model-tagged entities reproduces about 58% of the relations found from gold tags, with about 63% precision against them. Missed `GROUP` entities (136 predicted against 213 gold in test) are the main cause. These counts are small (102 reference relations) and the reference relations are themselves automatic, so treat them as indicative. For the 120 AnnoCTR reports the gold-tag graph remains the more accurate one.

## Semantic Relation Verification

The V1 NLI and Hybrid outputs are preserved as baselines. The separate,
deterministic V2 rule verifier and its conservative decision policy are
documented in [`docs/semantic_verifier_v2_implementation.md`](docs/semantic_verifier_v2_implementation.md).

## Data Tracking

The full AnnoCTR corpus remains in its upstream repository and is not copied into this repository. Selected AZERG input splits and the three derived knowledge-graph JSON files are tracked here. Generated predictions, annotations, audits, and other intermediate outputs are selectively excluded; do not assume that every file under `data/` should be committed. Check redistribution terms before sharing any dataset files.
