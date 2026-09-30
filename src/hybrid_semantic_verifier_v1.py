import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_NLI_INPUT = Path("data/semantic_verifier_v1_zero_shot.jsonl")
DEFAULT_V6_INPUT = Path("data/annoctr_relation_predictions_validated_v6.jsonl")
DEFAULT_OUTPUT = Path("data/semantic_verifier_v1_hybrid.jsonl")

# These are an explicit experimental policy, not learned or calibrated weights.
NLI_WEIGHT = 1.0
STRUCTURAL_WEIGHT = 0.5
EVIDENCE_WEIGHT = 0.25
CONTEXT_WEIGHT = 0.5
SUPPORTED_MINIMUM = 0.5
UNSUPPORTED_MAXIMUM = -0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic hybrid semantic verifier baseline."
    )
    parser.add_argument("--nli-input", type=Path, default=DEFAULT_NLI_INPUT)
    parser.add_argument("--v6-input", type=Path, default=DEFAULT_V6_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_key(record: dict) -> Tuple[object, ...]:
    return (
        record.get("document"),
        record.get("segment"),
        record.get("source", record.get("source_mention")),
        record.get("relation", record.get("predicted_relation")),
        record.get("target", record.get("target_mention")),
    )


def first_value(record: dict, fallback: dict, *names: str):
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
        if name in fallback and fallback[name] is not None:
            return fallback[name]
    return None


def bool_value(value: object) -> bool:
    return value is True


def relation_argument_features(record: dict, fallback: dict) -> Tuple[str, str]:
    arguments = first_value(record, fallback, "relation_arguments_v2") or {}
    direction = str(arguments.get("direction", "unknown")).lower()
    structure = str(arguments.get("argument_structure", "none")).upper()
    return direction, structure


def evidence_quality(record: dict, fallback: dict) -> Tuple[Optional[str], str]:
    text = record.get("evidence_text")
    quality = record.get("evidence_quality")
    if isinstance(text, str) and text.strip():
        return text.strip(), quality or "sentence"

    sentence = first_value(record, fallback, "evidence_sentence")
    if isinstance(sentence, str) and sentence.strip():
        return sentence.strip(), "sentence"

    segment = first_value(record, fallback, "text")
    if isinstance(segment, str) and segment.strip():
        return segment.strip(), "segment_fallback"

    return None, "missing"


def build_signals(record: dict, fallback: dict) -> dict:
    direction, structure = relation_argument_features(record, fallback)
    source_argument = bool_value(first_value(record, fallback, "source_argument"))
    target_argument = bool_value(first_value(record, fallback, "target_argument"))
    predicate_status = str(
        first_value(record, fallback, "predicate_status") or "UNKNOWN"
    ).upper()
    type_compatible = first_value(record, fallback, "type_compatible")

    positive = []
    negative = []
    structural_points = 0.0

    if source_argument:
        positive.append("source_argument")
        structural_points += 1
    if target_argument:
        positive.append("target_argument")
        structural_points += 1
    if predicate_status in {"FORWARD", "PASSIVE_FORWARD"}:
        positive.append("predicate_support")
        structural_points += 1
    elif predicate_status in {"NO_PREDICATE", "UNKNOWN", "NO_SENTENCE"}:
        negative.append(f"predicate_{predicate_status.lower()}")
        structural_points -= 1
    if direction == "forward" or "SOURCE_SUBJECT_TARGET_OBJECT" in structure:
        positive.append("forward_direction")
        structural_points += 1
    elif direction == "reverse" or "TARGET_SUBJECT_SOURCE_OBJECT" in structure:
        negative.append("reverse_direction")
        structural_points -= 1
    if type_compatible is True:
        positive.append("type_compatible")
        structural_points += 0.5
    elif type_compatible is False:
        negative.append("type_incompatible")
        structural_points -= 0.5

    risk_fields = (
        ("negated", "negation"),
        ("comparison_context", "comparison_context"),
        ("reporting_context", "reporting_context"),
        ("table_context", "table_context"),
        ("uncertainty_context", "uncertainty_context"),
    )
    risks = {}
    context_penalty = 0.0
    for field, label in risk_fields:
        active = bool_value(first_value(record, fallback, field))
        risks[field] = active
        if active:
            negative.append(label)
            context_penalty += 1.0

    return {
        "direction": direction,
        "argument_structure": structure,
        "source_argument": source_argument,
        "target_argument": target_argument,
        "predicate_status": predicate_status,
        "type_compatible": type_compatible,
        "structural_points": structural_points,
        "positive_signals": positive,
        "negative_signals": negative,
        "context_penalty_points": context_penalty,
        "risks": risks,
    }


def score_record(record: dict, fallback: dict) -> dict:
    evidence_text, quality = evidence_quality(record, fallback)
    entailment = record.get("entailment_score")
    contradiction = record.get("contradiction_score")
    neutral = record.get("neutral_score")
    nli_prediction = record.get("nli_prediction")
    signals = build_signals(record, fallback)

    scores_available = all(isinstance(value, (int, float)) for value in (entailment, contradiction, neutral))
    nli_score = float(entailment) - float(contradiction) if scores_available else None
    evidence_score = {"sentence": 1.0, "segment_fallback": 0.25, "missing": 0.0}[quality]
    structural_score = signals["structural_points"] / 4.5
    context_penalty = signals["context_penalty_points"]
    hybrid_score = None
    if nli_score is not None:
        hybrid_score = (
            NLI_WEIGHT * nli_score
            + STRUCTURAL_WEIGHT * structural_score
            + EVIDENCE_WEIGHT * evidence_score
            - CONTEXT_WEIGHT * context_penalty
        )

    direction_supported = (
        signals["direction"] == "forward"
        or "SOURCE_SUBJECT_TARGET_OBJECT" in signals["argument_structure"]
    )
    reverse_direction = (
        signals["direction"] == "reverse"
        or "TARGET_SUBJECT_SOURCE_OBJECT" in signals["argument_structure"]
    )
    predicate_supported = signals["predicate_status"] in {"FORWARD", "PASSIVE_FORWARD"}
    coordination_risk = "conj" in str(record.get("semantic_analysis", "")).lower()
    blocking_risks = signals["risks"]["negated"] or signals["risks"]["table_context"]
    blocking_risks = blocking_risks or signals["risks"]["reporting_context"]
    blocking_risks = blocking_risks or signals["risks"]["comparison_context"]

    if signals["source_argument"] and signals["target_argument"] and predicate_supported:
        argument_resolution = "resolved"
    elif signals["source_argument"] or signals["target_argument"]:
        argument_resolution = "partial"
    else:
        argument_resolution = "unresolved"

    if quality == "missing" or nli_score is None:
        status = "UNCERTAIN"
        reason = "Evidence or raw NLI scores are missing."
    elif reverse_direction:
        status = "UNSUPPORTED"
        reason = "The available argument diagnostics indicate the candidate direction is reversed."
    elif blocking_risks and nli_prediction == "entailment":
        status = "UNCERTAIN"
        reason = "NLI entailment is present, but a context-safety diagnostic blocks acceptance."
    elif hybrid_score >= SUPPORTED_MINIMUM and nli_prediction == "entailment" and quality == "sentence":
        status = "SUPPORTED"
        reason = "NLI entailment agrees with positive structural evidence and sentence-level evidence."
    elif hybrid_score <= UNSUPPORTED_MAXIMUM and nli_prediction == "contradiction":
        status = "UNSUPPORTED"
        reason = "NLI contradiction agrees with the combined structural and evidence signals."
    else:
        status = "UNCERTAIN"
        reason = "The combined semantic and structural evidence is insufficient for a decisive status."

    diagnostics = {
        "evidence_available": evidence_text is not None,
        "sentence_level_evidence": quality == "sentence",
        "source_supported": signals["source_argument"],
        "target_supported": signals["target_argument"],
        "predicate_supported": predicate_supported,
        "direction_supported": direction_supported and not reverse_direction,
        "type_compatible": signals["type_compatible"],
        "coordination_risk": coordination_risk,
        "reporting_risk": signals["risks"]["reporting_context"],
        "comparison_risk": signals["risks"]["comparison_context"],
        "table_risk": signals["risks"]["table_context"],
        "negation_risk": signals["risks"]["negated"],
        "argument_resolution": argument_resolution,
    }
    components = {
        "nli_score": nli_score,
        "entailment_score": entailment,
        "contradiction_score": contradiction,
        "neutral_score": neutral,
        "structural_score": structural_score,
        "evidence_quality_score": evidence_score,
        "context_penalty": context_penalty,
        "positive_signals": signals["positive_signals"],
        "negative_signals": signals["negative_signals"],
    }
    return {
        "hybrid_status": status,
        "hybrid_score": hybrid_score,
        "hybrid_components": components,
        "hybrid_diagnostics": diagnostics,
        "decision_reason": reason,
        "verifier_version": "hybrid_v1",
    }


def fallback_index(records: Iterable[dict]) -> Dict[Tuple[object, ...], dict]:
    return {record_key(record): record for record in records}


def average(values: List[float]) -> str:
    return f"{sum(values) / len(values):.4f}" if values else "n/a"


def print_report(records: List[dict]) -> None:
    total = len(records)
    status_counts = Counter(record["hybrid_status"] for record in records)
    print(f"Total records: {total}")
    for status in ("SUPPORTED", "UNSUPPORTED", "UNCERTAIN"):
        count = status_counts[status]
        percentage = 100 * count / total if total else 0
        print(f"{status}: {count} ({percentage:.2f}%)")
    print("NLI prediction distribution:", dict(Counter(record.get("nli_prediction") for record in records)))

    by_relation = defaultdict(Counter)
    by_v6 = defaultdict(Counter)
    by_quality = Counter(record.get("evidence_quality") for record in records)
    entailment = defaultdict(list)
    contradiction = defaultdict(list)
    for record in records:
        by_relation[record.get("relation")][record["hybrid_status"]] += 1
        by_v6[record.get("v6_status", record.get("status"))][record["hybrid_status"]] += 1
        if isinstance(record.get("entailment_score"), (int, float)):
            entailment[record["hybrid_status"]].append(record["entailment_score"])
        if isinstance(record.get("contradiction_score"), (int, float)):
            contradiction[record["hybrid_status"]].append(record["contradiction_score"])

    print("Evidence-quality distribution:", dict(by_quality))
    print("Hybrid status by relation:", {key: dict(value) for key, value in by_relation.items()})
    print("Hybrid status by V6 status:", {key: dict(value) for key, value in by_v6.items()})
    for status in ("SUPPORTED", "UNSUPPORTED", "UNCERTAIN"):
        print(f"Average entailment, {status}: {average(entailment[status])}")
        print(f"Average contradiction, {status}: {average(contradiction[status])}")

    top = sorted(
        (record for record in records if record["hybrid_status"] == "SUPPORTED"),
        key=lambda record: record["hybrid_score"] if record["hybrid_score"] is not None else float("-inf"),
        reverse=True,
    )[:20]
    print("Top 20 SUPPORTED candidates:")
    for rank, record in enumerate(top, start=1):
        print(json.dumps({
            "rank": rank,
            "source": record.get("source"),
            "relation": record.get("relation"),
            "target": record.get("target"),
            "evidence": record.get("evidence_text"),
            "hybrid_score": record.get("hybrid_score"),
            "entailment_score": record.get("entailment_score"),
            "contradiction_score": record.get("contradiction_score"),
            "decision_reason": record.get("decision_reason"),
        }, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if not args.nli_input.exists():
        raise FileNotFoundError(
            f"NLI input not found: {args.nli_input}. Run semantic_verifier_v1_zero_shot.py first."
        )
    nli_records = load_jsonl(args.nli_input)
    v6_records = load_jsonl(args.v6_input) if args.v6_input.exists() else []
    v6_by_key = fallback_index(v6_records)

    output_records = []
    for record in nli_records:
        fallback = v6_by_key.get(record_key(record), {})
        enriched = dict(record)
        enriched.update(score_record(record, fallback))
        output_records.append(enriched)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print_report(output_records)
    print("Hybrid Semantic Verifier V1 completed.")
    print("Evaluation requires frozen independent human labels.")


if __name__ == "__main__":
    main()