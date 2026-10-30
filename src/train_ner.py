"""Fine-tune a transformer for cybersecurity NER on AnnoCTR and evaluate it.

Trains on AnnoCTR train, picks the best epoch on dev, reports entity-level precision/recall/F1
(exact span + type match) on dev and test, and saves the model, metrics and test predictions.

Run from the cti_graph_rag directory:
    python src/train_ner.py --model distilroberta-base --epochs 4
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:  # pasted into a notebook cell
    ROOT = Path.cwd()
CORPUS = ROOT.parent / "anno-ctr-lrec-coling-2024" / "AnnoCTR" / "ner_json"
MODEL_DIR = ROOT / "models" / "ner"
METRICS_OUT = ROOT / "data" / "ner_eval.json"
PRED_OUT = ROOT / "data" / "ner_predictions_test.jsonl"
MAX_LEN = 192
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(split):
    with open(CORPUS / f"{split}.json", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def spans(tags):
    """Set of (start, end, type) entity spans from a BIO tag list."""
    out, start, etype = set(), None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag.startswith("I-") and start is not None and tag[2:] == etype:
            continue
        if start is not None:
            out.add((start, i, etype))
            start, etype = None, None
        if tag.startswith(("B-", "I-")):
            start, etype = i, tag[2:]
    return out


def score(gold_seqs, pred_seqs):
    """Entity-level exact-match precision/recall/F1, overall and per type."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for g, p in zip(gold_seqs, pred_seqs):
        gs, ps = spans(g), spans(p)
        for s in gs & ps:
            tp[s[2]] += 1
        for s in ps - gs:
            fp[s[2]] += 1
        for s in gs - ps:
            fn[s[2]] += 1

    def prf(t, f_p, f_n):
        p = t / (t + f_p) if t + f_p else 0.0
        r = t / (t + f_n) if t + f_n else 0.0
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": t + f_n}

    types = sorted(set(tp) | set(fp) | set(fn))
    per_type = {t: prf(tp[t], fp[t], fn[t]) for t in types}
    micro = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    macro_f1 = sum(v["f1"] for v in per_type.values()) / max(len(per_type), 1)
    return {"micro": micro, "macro_f1": macro_f1, "per_type": per_type}


def encode(tokenizer, records, label2id):
    enc = tokenizer([r["tokens"] for r in records], is_split_into_words=True, truncation=True,
                    max_length=MAX_LEN)
    items = []
    for i, r in enumerate(records):
        word_ids, labels, prev = enc.word_ids(i), [], None
        for w in word_ids:
            labels.append(-100 if w is None or w == prev else label2id[r["all_tags"][w]])
            prev = w
        items.append({"input_ids": enc["input_ids"][i], "labels": labels, "word_ids": word_ids})
    return items


def collate(batch, pad_id):
    """Pad a batch and move it to the training device."""
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    lab = torch.full((len(batch), n), -100)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k], mask[i, :k], lab[i, :k] = torch.tensor(b["input_ids"]), 1, torch.tensor(b["labels"])
    return ids.to(DEVICE), mask.to(DEVICE), lab.to(DEVICE)


@torch.no_grad()
def predict(model, items, records, id2label, pad_id, batch_size=64):
    model.eval()
    preds = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids, mask, _ = collate([items[i] for i in idx], pad_id)
        out = model(input_ids=ids, attention_mask=mask).logits.argmax(-1)
        for row, i in enumerate(idx):
            tags, prev = ["O"] * len(records[i]["tokens"]), None
            row_pred = out[row].tolist()
            for pos, w in enumerate(items[i]["word_ids"]):
                if w is not None and w != prev:
                    tags[w] = id2label[row_pred[pos]]
                prev = w
            preds[i] = tags
    return preds


def fix_bio(tags):
    """Turn stray I- tags (no matching B-/I- before them) into B- tags."""
    out, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            t = "B-" + t[2:]
        out.append(t)
        prev = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0, help="use only N training sentences (smoke test)")
    ap.add_argument("--corpus", default=None, help="folder with train.json, dev.json, test.json")
    ap.add_argument("--out", default=None, help="output folder for model, metrics and predictions")
    args, _ = ap.parse_known_args()  # ignores extra notebook kernel arguments

    global CORPUS, MODEL_DIR, METRICS_OUT, PRED_OUT
    if args.corpus:
        CORPUS = Path(args.corpus)
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        MODEL_DIR, METRICS_OUT, PRED_OUT = (out_dir / "ner_model", out_dir / "ner_eval.json",
                                            out_dir / "ner_predictions_test.jsonl")
    print("device:", DEVICE)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(torch.get_num_threads())

    train, dev, test = load("train"), load("dev"), load("test")
    if args.limit:
        train, dev, test = train[:args.limit], dev[:args.limit // 2], test[:args.limit // 2]
    labels = sorted({t for r in train + dev + test for t in r["all_tags"]}, key=lambda t: (t != "O", t))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=len(labels),
                                                            id2label=id2label, label2id=label2id).to(DEVICE)
    tr, dv, te = (encode(tokenizer, d, label2id) for d in (train, dev, test))
    trunc = sum(1 for items, recs in ((tr, train), (dv, dev), (te, test)) for it, r in zip(items, recs)
                if len({w for w in it["word_ids"] if w is not None}) < len(r["tokens"]))
    print(f"train {len(tr)}  dev {len(dv)}  test {len(te)}  labels {len(labels)}  truncated sentences {trunc}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(tr) + args.batch_size - 1) // args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    pad = tokenizer.pad_token_id

    best_f1, history, t0 = -1.0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(tr)))
        random.shuffle(order)
        total = 0.0
        for n, s in enumerate(range(0, len(order), args.batch_size), 1):
            ids, mask, lab = collate([tr[i] for i in order[s:s + args.batch_size]], pad)
            loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total += loss.item()
            if n % 50 == 0:
                print(f"  epoch {epoch} step {n} loss {total / n:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        dev_pred = [fix_bio(p) for p in predict(model, dv, dev, id2label, pad)]
        dev_m = score([r["all_tags"] for r in dev], dev_pred)
        history.append({"epoch": epoch, "train_loss": total / n, "dev_micro_f1": dev_m["micro"]["f1"]})
        print(f"epoch {epoch}: loss {total / n:.4f}  dev micro-F1 {dev_m['micro']['f1']:.4f}", flush=True)
        if dev_m["micro"]["f1"] > best_f1:
            best_f1 = dev_m["micro"]["f1"]
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    # plain PyTorch weights + label map, for loading without save_pretrained
    torch.save(model.state_dict(), MODEL_DIR / "ner_model.pth")
    (MODEL_DIR / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    dev_pred = [fix_bio(p) for p in predict(model, dv, dev, id2label, pad)]
    test_pred = [fix_bio(p) for p in predict(model, te, test, id2label, pad)]
    result = {
        "model": args.model, "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "train_sentences": len(tr), "history": history,
        "dev": score([r["all_tags"] for r in dev], dev_pred),
        "test": score([r["all_tags"] for r in test], test_pred),
        "metric": "entity-level exact match (span and type)",
    }
    METRICS_OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(PRED_OUT, "w", encoding="utf-8") as f:
        for r, p in zip(test, test_pred):
            f.write(json.dumps({"id": r["id"], "tokens": r["tokens"], "gold": r["all_tags"], "pred": p},
                               ensure_ascii=False) + "\n")

    print(f"\nTEST micro P/R/F1: {result['test']['micro']['precision']:.3f} / "
          f"{result['test']['micro']['recall']:.3f} / {result['test']['micro']['f1']:.3f}   "
          f"macro-F1 {result['test']['macro_f1']:.3f}")
    for t, v in result["test"]["per_type"].items():
        print(f"  {t:24s} P {v['precision']:.3f}  R {v['recall']:.3f}  F1 {v['f1']:.3f}  n={v['support']}")


if __name__ == "__main__":
    main()
