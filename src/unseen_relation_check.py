"""End-to-end check of the relation extractor on reports it was never tuned on.

AnnoCTR's `train_ext` split has 280 reports outside the 120 gold reports, but they carry only general
tags (ORG, LOC, DATE, SECTOR, code), not malware/group/technique. So this tags a random sample of
those reports with the trained NER model, runs the rule-based relation extractor (v2) on the
predicted entities, and writes a sample of relations for review.

Reads:  AnnoCTR ner_json/train_ext.json, models/ner/ner_model.pth, models/ner/labels.json
Writes: data/unseen_check_relations.jsonl         every relation found in the sampled reports
        data/unseen_check_audit_template.csv      stratified sample for review (labels blank)

Run from the cti_graph_rag directory:  python src/unseen_relation_check.py
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, RobertaConfig, RobertaForTokenClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict_ner  # noqa: E402  (weights check and tokenizer location)
import train_ner as tn  # noqa: E402
from extract_relations import find_relations  # noqa: E402
from preprocess_reports import CVE_RE, char_offsets  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT.parent / "anno-ctr-lrec-coling-2024" / "AnnoCTR" / "ner_json"
SEED, SEED_ROUND2, SEED_ROUND3, N_DOCS = 7, 11, 23, 60
ROUND = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # 2 = fresh sample of reports not used in round 1
SUFFIX = "" if ROUND == 1 else f"_round{ROUND}"
SAMPLE_PER_RELATION = {"EXPLOITS": 10, "USES_TECHNIQUE": 15, "USES": 25, "TARGETS": 30}


def gold_documents():
    docs = set()
    for split in ("train", "dev", "test"):
        with open(CORPUS / f"{split}.json", encoding="utf-8") as f:
            docs |= {json.loads(l)["id"].rsplit("__s", 1)[0] for l in f}
    return docs


def main():
    tn.DEVICE = torch.device("cpu")
    tn.CORPUS = CORPUS
    seen = gold_documents()
    by_doc = defaultdict(list)
    with open(CORPUS / "train_ext.json", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            doc, seg = r["id"].rsplit("__s", 1)
            if doc not in seen:
                r["segment"] = int(seg)
                by_doc[doc].append(r)
    def choose(pool, seed):
        pool = sorted(pool)
        random.Random(seed).shuffle(pool)
        # keep the vendor mix of the pool: zscaler/proofpoint dominate, so take every small-vendor doc first
        small = [d for d in pool if d.split("_")[0] not in ("zscaler", "proofpoint")]
        big = [d for d in pool if d.split("_")[0] in ("zscaler", "proofpoint")]
        return (small + big)[:N_DOCS]

    docs = sorted(by_doc)
    chosen = choose(docs, SEED)
    if ROUND >= 2:    # a fresh sample: different reports from round 1, so rules tuned on round 1 are tested unseen
        used = set(chosen)
        chosen = choose([d for d in docs if d not in used], SEED_ROUND2)
    if ROUND == 3:    # and again: reports not used in rounds 1 or 2
        used |= set(chosen)
        chosen = choose([d for d in docs if d not in used], SEED_ROUND3)
    records = [r for d in chosen for r in sorted(by_doc[d], key=lambda r: r["segment"])]
    print(f"unseen reports in pool {len(docs)}; sampled {len(chosen)}; sentences {len(records)}", flush=True)

    predict_ner.require_weights()
    labels = json.loads((ROOT / "models" / "ner" / "labels.json").read_text(encoding="utf-8"))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(predict_ner.TOKENIZER, add_prefix_space=True)
    cfg = RobertaConfig(num_labels=len(labels), id2label=id2label, label2id=label2id, vocab_size=50265,
                        max_position_embeddings=514, type_vocab_size=1, pad_token_id=1, bos_token_id=0,
                        eos_token_id=2, layer_norm_eps=1e-5)
    model = RobertaForTokenClassification(cfg)
    model.load_state_dict(torch.load(ROOT / "models" / "ner" / "ner_model.pth", map_location="cpu"))

    for r in records:  # the tagger only needs tokens; its gold-style tags are not used
        r["all_tags"] = ["O"] * len(r["tokens"])
    items = tn.encode(tokenizer, records, label2id)
    preds = [tn.fix_bio(p) for p in tn.predict(model, items, records, id2label, tokenizer.pad_token_id)]
    print("tagged", flush=True)

    found, type_counts = [], Counter()
    for r, tags in zip(records, preds):
        offs = char_offsets(r["text"], r["tokens"])
        ents = []
        for s, e, t in sorted(tn.spans(tags)):
            a, b = offs[s], offs[e - 1]
            if a and b:
                ents.append({"text": " ".join(r["tokens"][s:e]), "type": t, "char_start": a[0], "char_end": b[1]})
        for m in CVE_RE.finditer(r["text"]):
            if not any(e["char_start"] < m.end() and m.start() < e["char_end"] for e in ents):
                ents.append({"text": m[0], "type": "CVE", "char_start": m.start(), "char_end": m.end()})
        type_counts.update(e["type"] for e in ents)
        for rel, h, t, trig, direction, conf in find_relations({"text": r["text"], "entities": ents}, {}):
            found.append({"document": r["id"].rsplit("__s", 1)[0], "segment": r["segment"], "relation": rel,
                          "head_mention": h["text"], "head_type": h["type"], "tail_mention": t["text"],
                          "tail_type": t["type"], "trigger": trig, "direction": direction, "confidence": conf,
                          "evidence_text": r["text"]})
    with open(ROOT / "data" / f"unseen_check_relations{SUFFIX}.jsonl", "w", encoding="utf-8") as f:
        for x in found:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("predicted entity types:", dict(type_counts))
    print("relations found:", Counter(x["relation"] for x in found), "| confidence:", Counter(x["confidence"] for x in found))

    rng = random.Random(SEED)
    sample = []
    for rel, n in SAMPLE_PER_RELATION.items():
        pool = [x for x in found if x["relation"] == rel]
        sample += rng.sample(pool, min(n, len(pool)))
    cols = ["audit_id", "document", "segment", "relation", "head_mention", "head_type", "tail_mention", "tail_type",
            "trigger", "direction", "confidence", "evidence_text", "human_label", "error_category", "rationale",
            "reviewer", "reviewed_at"]
    with open(ROOT / "data" / f"unseen_check_audit_template{SUFFIX}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for i, x in enumerate(sample, 1):
            w.writerow({"audit_id": f"UNS-{i:03d}", **{k: x[k] for k in cols if k in x}})
    print(f"wrote {len(sample)} sampled relations to data/unseen_check_audit_template{SUFFIX}.csv")


if __name__ == "__main__":
    main()
