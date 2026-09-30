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

## Data Tracking

The full AnnoCTR corpus remains in its upstream repository and is not copied into this repository. Selected AZERG input splits and the three derived knowledge-graph JSON files are tracked here. Generated predictions, annotations, audits, and other intermediate outputs are selectively excluded; do not assume that every file under `data/` should be committed. Check redistribution terms before sharing any dataset files.
