import json
from collections import Counter, defaultdict
from pathlib import Path


AUDIT_PATH = Path("data/relation_random_audit_200.jsonl")
NLI_PATH = Path("data/semantic_verifier_v1_zero_shot.jsonl")
V6_PATH = Path("data/annoctr_relation_predictions_validated_v6.jsonl")
HYBRID_PATH = Path("data/semantic_verifier_v1_hybrid.jsonl")
OUTPUT_PATH = Path("data/hybrid_audit_evaluation_200.json")

NLI_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99)
TRUE_IDS = {14, 125, 127, 160, 179}
UNCERTAIN_IDS = {56, 88, 95, 97}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def join_key(record):
    return (
        record.get("document"),
        record.get("segment"),
        record.get("source", record.get("source_mention")),
        record.get("relation", record.get("predicted_relation")),
        record.get("target", record.get("target_mention")),
    )


def index_records(records):
    indexed = defaultdict(list)
    for record in records:
        key = join_key(record)
        indexed[key].append(record)
    return indexed


def join_audit(audit_records, other_records, name):
    audit_index = index_records(audit_records)
    duplicate_audit_keys = [key for key, records in audit_index.items() if len(records) != 1]
    if duplicate_audit_keys:
        raise ValueError(f"Audit contains duplicate join keys: {duplicate_audit_keys[:3]}")
    other_index = index_records(other_records)
    joined = []
    invalid = []
    for audit in audit_records:
        key = join_key(audit)
        matches = other_index.get(key, [])
        if len(matches) != 1:
            invalid.append((key, len(matches)))
        else:
            joined.append((audit, matches[0]))
    if invalid:
        raise ValueError(
            f"{name} did not join exactly once for {len(invalid)} audit keys; "
            f"first: {invalid[0]}"
        )
    if len(joined) != len(audit_records):
        raise ValueError(f"{name} did not join exactly once for every audit record")
    return joined


def assign_human_label(audit_id):
    if audit_id in TRUE_IDS:
        return "TRUE"
    if audit_id in UNCERTAIN_IDS:
        return "UNCERTAIN"
    return "FALSE"


def binary_metrics(labels, predictions):
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions))
    tn = sum(not label and not prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    coverage = sum(predictions) / len(labels) if labels else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "coverage": coverage,
        "n_evaluated": len(labels),
    }


def evaluate_join(joined, prediction):
    evaluated = [
        (audit, output)
        for audit, output in joined
        if audit["human_label"] != "UNCERTAIN"
    ]
    labels = [audit["human_label"] == "TRUE" for audit, _ in evaluated]
    predictions = [prediction(output) for _, output in evaluated]
    return binary_metrics(labels, predictions)


def raw_record(audit, output, status=None):
    return {
        "audit_id": audit["audit_id"],
        "source": audit.get("source"),
        "relation": audit.get("relation"),
        "target": audit.get("target"),
        "human_label": audit["human_label"],
        "error_category": audit.get("error_category", ""),
        "hybrid_status": output.get("hybrid_status"),
        "hybrid_score": output.get("hybrid_score"),
        "entailment_score": output.get("entailment_score"),
        "contradiction_score": output.get("contradiction_score"),
        "neutral_score": output.get("neutral_score"),
        "nli_prediction": output.get("nli_prediction"),
        "v6_status": output.get("v6_status", output.get("status")),
        "predicate_status": output.get("predicate_status"),
        "evidence": output.get("evidence_text", output.get("evidence_sentence", output.get("text"))),
        "decision_reason": output.get("decision_reason"),
    }


def roc_auc(scores, labels):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def print_metrics(name, metrics):
    print(
        f"{name}: TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} "
        f"FN={metrics['fn']} Precision={metrics['precision']:.4f} "
        f"Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
        f"Accuracy={metrics['accuracy']:.4f} Coverage={metrics['coverage']:.4f}"
    )


def main():
    for path in (AUDIT_PATH, NLI_PATH, V6_PATH, HYBRID_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    audit_records = load_jsonl(AUDIT_PATH)
    if len(audit_records) != 200:
        raise ValueError(f"Expected 200 audit records, found {len(audit_records)}")

    for audit in audit_records:
        audit["human_label"] = assign_human_label(int(audit["audit_id"]))

    if Counter(audit["human_label"] for audit in audit_records) != Counter(
        {"TRUE": 5, "FALSE": 191, "UNCERTAIN": 4}
    ):
        raise ValueError("Audit label distribution does not match the declared pilot labels")

    nli_joined = join_audit(audit_records, load_jsonl(NLI_PATH), "zero-shot NLI")
    v6_joined = join_audit(audit_records, load_jsonl(V6_PATH), "V6")
    hybrid_joined = join_audit(audit_records, load_jsonl(HYBRID_PATH), "hybrid")

    for audit, _ in nli_joined:
        audit["human_label"] = assign_human_label(int(audit["audit_id"]))

    systems = {}
    systems["V6"] = evaluate_join(v6_joined, lambda record: record.get("status") == "SUPPORTED")
    systems["NLI entailment prediction"] = evaluate_join(
        nli_joined, lambda record: record.get("nli_prediction") == "entailment"
    )
    for threshold in NLI_THRESHOLDS:
        name = f"NLI threshold {threshold:.2f}"
        systems[name] = evaluate_join(
            nli_joined,
            lambda record, threshold=threshold: (
                isinstance(record.get("entailment_score"), (int, float))
                and record["entailment_score"] >= threshold
            ),
        )
    systems["Hybrid V1"] = evaluate_join(
        hybrid_joined, lambda record: record.get("hybrid_status") == "SUPPORTED"
    )

    threshold_results = {
        f"{threshold:.2f}": systems[f"NLI threshold {threshold:.2f}"]
        for threshold in NLI_THRESHOLDS
    }

    hybrid_false_positives = [
        raw_record(audit, output)
        for audit, output in hybrid_joined
        if audit["human_label"] == "FALSE" and output.get("hybrid_status") == "SUPPORTED"
    ]
    hybrid_true_results = [
        raw_record(audit, output)
        for audit, output in hybrid_joined
        if audit["human_label"] == "TRUE"
    ]
    fp_by_category = Counter(record.get("error_category") or "UNKNOWN" for record in hybrid_false_positives)

    evaluated_nli = [
        (audit, output)
        for audit, output in nli_joined
        if audit["human_label"] != "UNCERTAIN"
    ]
    evaluated_hybrid = [
        (audit, output)
        for audit, output in hybrid_joined
        if audit["human_label"] != "UNCERTAIN"
    ]
    labels = [audit["human_label"] == "TRUE" for audit, _ in evaluated_nli]
    auc = {
        "entailment_score": roc_auc(
            [output.get("entailment_score", 0.0) for _, output in evaluated_nli], labels
        ),
        "entailment_minus_contradiction": roc_auc(
            [
                output.get("entailment_score", 0.0) - output.get("contradiction_score", 0.0)
                for _, output in evaluated_nli
            ],
            labels,
        ),
        "hybrid_score": roc_auc(
            [output.get("hybrid_score", 0.0) for _, output in evaluated_hybrid], labels
        ),
    }

    result = {
        "evaluation_type": "pilot_human_audit",
        "n_total": 200,
        "n_true": 5,
        "n_false": 191,
        "n_uncertain": 4,
        "uncertain_excluded": True,
        "n_evaluated_binary": 196,
        "join_key": ["document", "segment", "source", "relation", "target"],
        "joins_verified_exactly_once": True,
        "systems": systems,
        "nli_threshold_results": threshold_results,
        "auc": auc,
        "false_positive_examples": hybrid_false_positives,
        "false_positive_by_error_category": dict(fp_by_category),
        "true_example_results": hybrid_true_results,
        "methodological_notes": [
            "This is a pilot human audit, not a final independent gold evaluation.",
            "UNCERTAIN human examples were excluded from binary precision, recall, F1, and accuracy.",
            "No verifier thresholds were tuned using this audit.",
            "The provisional 201-300 annotations were not used.",
            "Raw model scores were preserved in the detailed error and true-example records.",
            "Pilot AUC estimates are unstable because the audit contains only 5 positive examples.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Joined exactly once: {len(audit_records)} audit records")
    print("Excluded uncertain examples: 4")
    print("\nMetrics:")
    print("System | Precision | Recall | F1 | Coverage")
    for name, metrics in systems.items():
        print(
            f"{name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['coverage']:.4f}"
        )

    print("\nDetailed metrics:")
    for name, metrics in systems.items():
        print_metrics(name, metrics)

    print("\nHybrid false positives:")
    for item in hybrid_false_positives:
        print(json.dumps(item, ensure_ascii=False))
    print("False positives by error category:", dict(fp_by_category))

    print("\nHybrid TRUE examples:")
    for item in hybrid_true_results:
        print(json.dumps(item, ensure_ascii=False))

    print("\nAUC:", auc)
    print("Pilot AUC estimates are unstable because the audit contains only 5 positive examples.")
    print(f"Output: {OUTPUT_PATH}")
    print("Pilot audit evaluation completed.")
    print("These results are not the final gold evaluation.")
    print("Independent reannotation and a larger frozen evaluation set are still required.")


if __name__ == "__main__":
    main()