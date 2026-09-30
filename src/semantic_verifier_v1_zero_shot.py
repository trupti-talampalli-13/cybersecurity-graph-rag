import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_INPUT = Path("data/annoctr_relation_predictions_validated_v6.jsonl")
DEFAULT_OUTPUT = Path("data/semantic_verifier_v1_zero_shot.jsonl")
DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

RELATION_TEMPLATES = {
    "attributed-to": "{source} is attributed to {target}.",
    "authored-by": "{source} is authored by {target}.",
    "communicates-with": "{source} communicates with {target}.",
    "delivers": "{source} delivers {target}.",
    "downloads": "{source} downloads {target}.",
    "drops": "{source} drops {target}.",
    "exfiltrates-to": "{source} exfiltrates data to {target}.",
    "exploits": "{source} exploits {target}.",
    "impersonates": "{source} impersonates {target}.",
    "indicates": "{source} indicates {target}.",
    "located-at": "{source} is located at {target}.",
    "originates-from": "{source} originates from {target}.",
    "owns": "{source} owns {target}.",
    "targets": "{source} targets {target}.",
    "uses": "{source} uses {target}.",
    "variant-of": "{source} is a variant of {target}.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot NLI over V6 relation candidates."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def load_records(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_evidence(record: dict) -> Tuple[str, str]:
    sentence = record.get("evidence_sentence")
    if isinstance(sentence, str) and sentence.strip():
        return sentence.strip(), "sentence"

    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip(), "segment_fallback"

    return "", "missing"


def build_hypothesis(record: dict) -> str:
    source = str(record.get("source", record.get("source_mention", ""))).strip()
    target = str(record.get("target", record.get("target_mention", ""))).strip()
    relation = str(record.get("relation", record.get("predicted_relation", ""))).strip()

    template = RELATION_TEMPLATES.get(relation)
    if template is None:
        raise ValueError(
            f"No fixed hypothesis template for relation {relation!r}. "
            "Add it explicitly before running the baseline."
        )
    return template.format(source=source, target=target)


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    if "entail" in normalized:
        return "entailment"
    if "contrad" in normalized:
        return "contradiction"
    if "neutral" in normalized:
        return "neutral"
    return normalized


def build_label_indices(config) -> Dict[str, int]:
    id_to_label = {int(index): label for index, label in config.id2label.items()}
    label_to_index = {
        normalize_label(label): index for index, label in id_to_label.items()
    }
    required = {"entailment", "contradiction", "neutral"}
    missing = required - label_to_index.keys()
    if missing:
        raise ValueError(
            f"Could not map NLI labels {sorted(missing)} from model config: "
            f"{id_to_label}"
        )
    return label_to_index


def iter_batches(items: List[Tuple[str, str]], batch_size: int) -> Iterable[List[Tuple[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_nli(
    pairs: List[Tuple[str, str]],
    tokenizer,
    model,
    label_indices: Dict[str, int],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> List[dict]:
    predictions = []
    model.eval()

    with torch.inference_mode():
        for batch in iter_batches(pairs, batch_size):
            premises, hypotheses = zip(*batch)
            encoded = tokenizer(
                list(premises),
                list(hypotheses),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)

            for row in probabilities.detach().cpu().tolist():
                scores = {
                    label: float(row[index])
                    for label, index in label_indices.items()
                }
                prediction = max(scores, key=scores.get)
                predictions.append(
                    {
                        "entailment_score": scores["entailment"],
                        "contradiction_score": scores["contradiction"],
                        "neutral_score": scores["neutral"],
                        "nli_prediction": prediction,
                    }
                )

    return predictions


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    records = load_records(args.input)
    prepared = []
    missing_indices = set()

    for index, record in enumerate(records):
        evidence_text, evidence_quality = select_evidence(record)
        hypothesis = build_hypothesis(record)
        prepared.append((evidence_text, evidence_quality, hypothesis))
        if evidence_quality == "missing":
            missing_indices.add(index)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    device = torch.device(device_name)

    print(f"Loading model: {args.model} (revision={args.revision})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.revision,
    ).to(device)
    label_indices = build_label_indices(model.config)
    resolved_revision = getattr(model.config, "_commit_hash", None) or args.revision

    pairs = [
        (evidence_text, hypothesis)
        for evidence_text, evidence_quality, hypothesis in prepared
        if evidence_quality != "missing"
    ]
    predictions = run_nli(
        pairs,
        tokenizer,
        model,
        label_indices,
        device,
        args.batch_size,
        args.max_length,
    )

    prediction_index = 0
    output_records = []
    for index, (record, prepared_record) in enumerate(zip(records, prepared)):
        evidence_text, evidence_quality, hypothesis = prepared_record
        output = dict(record)
        output.update(
            {
                "evidence_text": evidence_text or None,
                "evidence_quality": evidence_quality,
                "hypothesis": hypothesis,
                "nli_model": args.model,
                "nli_model_revision": resolved_revision,
                "nli_device": str(device),
                "score_type": "raw_model_score",
                "verifier_version": "semantic_evidence_verifier_v1_zero_shot",
            }
        )

        if index in missing_indices:
            output.update(
                {
                    "entailment_score": None,
                    "contradiction_score": None,
                    "neutral_score": None,
                    "nli_prediction": None,
                    "nli_skipped_reason": "missing_evidence",
                }
            )
        else:
            output.update(predictions[prediction_index])
            output["nli_skipped_reason"] = None
            prediction_index += 1

        output_records.append(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Input records: {len(records)}")
    print(f"NLI-inferred records: {len(records) - len(missing_indices)}")
    print(f"Missing-evidence records: {len(missing_indices)}")
    print(f"Output: {args.output}")
    print(f"Model revision: {resolved_revision}")
    print("No SUPPORTED/UNSUPPORTED/UNCERTAIN thresholds were applied.")


if __name__ == "__main__":
    main()