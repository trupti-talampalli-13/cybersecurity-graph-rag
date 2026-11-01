"""Tag AnnoCTR reports with the fine-tuned NER model and compare against the gold tags.

Loads models/ner/ner_model.pth (weights trained on Kaggle) and tags every sentence of the
train/dev/test splits. Writes:
  data/predicted_entities.jsonl   one record per sentence with predicted entity spans
  data/ner_vs_gold.json           entity-level scores per split (train scores are optimistic:
                                  the model saw those sentences while training)

Run from the cti_graph_rag directory:  python src/predict_ner.py
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, RobertaConfig, RobertaForTokenClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_ner as tn  # noqa: E402  (reuses encoding, prediction and scoring code)

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "ner" / "ner_model.pth"
LABELS = ROOT / "models" / "ner" / "labels.json"
# roberta-base tokenizer: a local copy if present, otherwise downloaded from the Hugging Face Hub
LOCAL_TOKENIZER = ROOT / "models" / "distilroberta-base"   # same BPE vocabulary as roberta-base
TOKENIZER = LOCAL_TOKENIZER if LOCAL_TOKENIZER.exists() else "roberta-base"


def require_weights():
    """Stop with an actionable message when the trained model files are missing."""
    missing = [p for p in (WEIGHTS, LABELS) if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing model files: " + ", ".join(str(p.relative_to(ROOT)) for p in missing) + ".\n"
            "The trained weights (about 500 MB) are not stored in Git. Get ner_model.pth and labels.json from the "
            "project owner, or train them with src/train_ner.py (see 'Trained NER Model' in the README), and put "
            "both in models/ner/.")
SPLITS = ["train", "dev", "test"]


def main():
    tn.DEVICE = torch.device("cpu")
    require_weights()
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, add_prefix_space=True)
    cfg = RobertaConfig(num_labels=len(labels), id2label=id2label, label2id=label2id,  # roberta-base sizes
                        vocab_size=50265, max_position_embeddings=514, type_vocab_size=1,
                        pad_token_id=1, bos_token_id=0, eos_token_id=2, layer_norm_eps=1e-5)
    model = RobertaForTokenClassification(cfg)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    print(f"loaded {WEIGHTS.name}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")

    summary = {}
    with open(ROOT / "data" / "predicted_entities.jsonl", "w", encoding="utf-8") as out:
        for split in SPLITS:
            tn.CORPUS = ROOT.parent / "anno-ctr-lrec-coling-2024" / "AnnoCTR" / "ner_json"
            records = tn.load(split)
            items = tn.encode(tokenizer, records, label2id)
            preds = [tn.fix_bio(p) for p in tn.predict(model, items, records, id2label, tokenizer.pad_token_id)]
            summary[split] = tn.score([r["all_tags"] for r in records], preds)
            m = summary[split]["micro"]
            print(f"{split:5s} sentences {len(records):5d}  P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}",
                  flush=True)
            for r, p in zip(records, preds):
                report_id, seg = r["id"].rsplit("__s", 1)
                ents = [{"text": " ".join(r["tokens"][s:e]), "type": t, "token_start": s, "token_end": e}
                        for s, e, t in sorted(tn.spans(p))]
                out.write(json.dumps({"report_id": report_id, "segment_id": int(seg), "split": split,
                                      "predicted_entities": ents}, ensure_ascii=False) + "\n")

    (ROOT / "data" / "ner_vs_gold.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote data/predicted_entities.jsonl and data/ner_vs_gold.json")


if __name__ == "__main__":
    main()
