"""How much does NER error hurt the knowledge graph?

Runs the rule-based relation extractor twice: once on the gold entity tags and once on the
entities predicted by the trained NER model, then compares entities and relations.
The gold-entity relations are the reference, so this measures how well the pipeline reproduces
them on text tagged by the model (not whether the relations themselves are correct).

Reads:  data/processed_segments.jsonl, data/predicted_entities.jsonl, AnnoCTR ner_json
Writes: data/ner_graph_comparison.json

Run from the cti_graph_rag directory:  python src/compare_gold_vs_predicted.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_relations import find_relations, norm  # noqa: E402
from preprocess_reports import CVE_RE, char_offsets  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NER_JSON = ROOT.parent / "anno-ctr-lrec-coling-2024" / "AnnoCTR" / "ner_json"


def load_tokens():
    tokens = {}
    for split in ("train", "dev", "test"):
        with open(NER_JSON / f"{split}.json", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                report, seg = r["id"].rsplit("__s", 1)
                tokens[(report, int(seg))] = r["tokens"]
    return tokens


def add_cves(text, entities):
    """Same CVE regex step as preprocessing, skipping spans that overlap an existing entity."""
    for m in CVE_RE.finditer(text):
        if not any(e["char_start"] is not None and e["char_start"] < m.end() and m.start() < e["char_end"]
                   for e in entities):
            entities.append({"text": m[0], "type": "CVE", "char_start": m.start(), "char_end": m.end()})
    return entities


def relation_keys(seg):
    keys = set()
    for rel, h, t, *_ in find_relations(seg, {}):
        keys.add((seg["report_id"], seg["segment_id"], rel, norm(h["text"]), norm(t["text"])))
    return keys


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "gold": n_gold, "predicted": n_pred, "matched": tp}


def main():
    tokens = load_tokens()
    predicted = {}
    with open(DATA / "predicted_entities.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            predicted[(r["report_id"], r["segment_id"])] = r["predicted_entities"]

    ent_gold, ent_pred = {}, {}
    rel_gold, rel_pred = {}, {}
    for split in ("dev", "test", "train"):
        ent_gold[split], ent_pred[split] = Counter(), Counter()
        rel_gold[split], rel_pred[split] = set(), set()

    with open(DATA / "processed_segments.jsonl", encoding="utf-8") as f:
        for line in f:
            seg = json.loads(line)
            key = (seg["report_id"], seg["segment_id"])
            if key not in predicted:
                continue
            split = seg["split"]
            toks = tokens[key]
            offs = char_offsets(seg["text"], toks)
            pred_ents = []
            for e in predicted[key]:
                a, b = offs[e["token_start"]], offs[e["token_end"] - 1]
                pred_ents.append({"text": e["text"], "type": e["type"],
                                  "char_start": a[0] if a else None, "char_end": b[1] if b else None})
            pred_ents = add_cves(seg["text"], pred_ents)
            pred_seg = {**seg, "entities": [e for e in pred_ents if e["char_start"] is not None]}

            for e in seg["entities"]:
                ent_gold[split][e["type"]] += 1
            for e in pred_ents:
                ent_pred[split][e["type"]] += 1
            rel_gold[split] |= relation_keys(seg)
            rel_pred[split] |= relation_keys(pred_seg)

    out = {"note": "relations from gold entities are the reference; train split is optimistic because "
                   "the model was trained on it", "entity_counts": {}, "relations": {}}
    for split in ("test", "dev", "train"):
        out["entity_counts"][split] = {"gold": dict(ent_gold[split]), "predicted": dict(ent_pred[split])}
        per = {}
        for rel in ("ALL", "USES", "USES_TECHNIQUE", "EXPLOITS", "TARGETS"):
            g = {k for k in rel_gold[split] if rel == "ALL" or k[2] == rel}
            p = {k for k in rel_pred[split] if rel == "ALL" or k[2] == rel}
            per[rel] = prf(len(g & p), len(p), len(g))
        out["relations"][split] = per
    (DATA / "ner_graph_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    for split in ("test", "dev", "train"):
        print(f"\n[{split}] relations reproduced from model-tagged text (reference = gold-entity relations)")
        for rel, v in out["relations"][split].items():
            print(f"  {rel:15s} gold {v['gold']:4d}  predicted {v['predicted']:4d}  matched {v['matched']:4d}  "
                  f"P {v['precision']:.2f}  R {v['recall']:.2f}  F1 {v['f1']:.2f}")
    print("\n[test] entity counts  gold -> predicted")
    for t in sorted(set(ent_gold["test"]) | set(ent_pred["test"])):
        print(f"  {t:24s} {ent_gold['test'][t]:5d} -> {ent_pred['test'][t]:5d}")


if __name__ == "__main__":
    main()
