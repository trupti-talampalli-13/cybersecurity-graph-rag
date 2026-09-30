# Semantic Evidence Verifier V1

## Project Context

This document specifies the next Person 1 component for the Graph-Augmented Multi-Hop CTI-QA project.

The existing pipeline is:

```text
AnnoCTR reports
-> NER
-> entity normalization
-> entity linking
-> relation candidate generation
-> type filtering
-> SVM relation proposal
-> argument extraction
-> V6 structural validation
```

The current relation pipeline is not reliable enough for direct knowledge-graph construction. A random semantic audit found 5 true, 191 false, and 4 uncertain predictions, giving 2.55% precision when uncertain examples are excluded.

The central methodological finding is:

> Entity co-occurrence is not evidence of a semantic relation.

Semantic Evidence Verifier V1 must therefore evaluate whether the evidence text entails the complete proposed relation:

```text
SOURCE --RELATION--> TARGET
```

It must not be another V7/V8/V9 rule-based validator.

## Current Input Schemas

### Relation Predictions With Argument Analysis

File:

```text
data/annoctr_relation_predictions_arguments_v2.jsonl
```

The file contains 5,534 records with these fields:

| Field | Meaning |
|---|---|
| `source_mention` | Proposed source entity mention |
| `source_type` | Source entity type |
| `target_mention` | Proposed target entity mention |
| `target_type` | Target entity type |
| `predicted_relation` | SVM-predicted relation |
| `text` | Original AnnoCTR segment text |
| `document` | Report identifier |
| `segment` | Segment identifier |
| `decision_score` | SVM decision score |
| `second_best_score` | Second-best class score |
| `margin` | Difference between top two SVM scores |
| `linguistic_analysis` | Dependency and token-level diagnostics |
| `relation_arguments_v2` | Extracted relation argument information |

Important observed values:

- `direction="unknown"`: 5,435 records
- `argument_structure="none"`: 5,435 records
- Forward direction: 42 records
- Reverse direction: 57 records

This confirms that V2 argument information is useful as an auxiliary feature, but cannot be treated as ground truth.

### V6 Validated Predictions

File:

```text
data/annoctr_relation_predictions_validated_v6.jsonl
```

V6 preserves the V2 fields and adds:

| Field | Meaning |
|---|---|
| `source` | Normalized source value |
| `target` | Normalized target value |
| `relation` | Relation used by V6 |
| `evidence_sentence` | Selected evidence sentence, when available |
| `status` | `SUPPORTED`, `PLAUSIBLE`, `UNSUPPORTED`, or `UNVERIFIED` |
| `predicate` | Detected predicate, when available |
| `predicate_status` | Predicate detection result |
| `source_argument` | Whether source was identified as an argument |
| `target_argument` | Whether target was identified as an argument |
| `negated` | Negation diagnostic |
| `comparison_context` | Comparison or similarity diagnostic |
| `reporting_context` | Reporting or observer-context diagnostic |
| `table_context` | Table, heading, or caption diagnostic |
| `uncertainty_context` | Uncertainty language diagnostic |
| `type_compatible` | Type-compatibility result |
| `validation_reasons` | Explanations for the V6 decision |
| `semantic_analysis` | V6 semantic result, currently stored as a string |

V6 statistics:

| Status or feature | Count |
|---|---:|
| `SUPPORTED` | 4 |
| `PLAUSIBLE` | 907 |
| `UNSUPPORTED` | 244 |
| `UNVERIFIED` | 4,379 |
| Missing `evidence_sentence` | 811 |
| `predicate_status="NO_PREDICATE"` | 3,424 |
| `predicate_status="UNKNOWN"` | 951 |

V1 must not assume that V6 successfully identified the grammatical arguments.

## V1 Objective

Given:

```text
SOURCE
RELATION
TARGET
EVIDENCE
```

predict one of:

```text
SUPPORTED
UNSUPPORTED
UNCERTAIN
```

The question is not whether the sentence is generally related to cybersecurity. The question is whether the sentence supports this exact proposition.

Example:

```text
Evidence: Leviathan uses web shells to maintain persistence.
Candidate: Leviathan --uses--> web shells
Expected: SUPPORTED
```

Counterexample:

```text
Evidence: NetWireRC is also used by APT threat actors.
Candidate: NetWireRC --uses--> APT threat actors
Expected: UNSUPPORTED
```

The evidence expresses the reverse direction.

## Recommended Architecture

```text
V6 candidate record
        |
        v
Evidence selection
        |
        v
Natural-language relation hypothesis
        |
        v
Zero-shot NLI model
        |
        +--> entailment score
        +--> contradiction score
        +--> neutral score
        |
        +--> V6 and V2 diagnostic features
        |
        v
SUPPORTED / UNSUPPORTED / UNCERTAIN
```

The first version should be a zero-shot NLI inference baseline. It should not fine-tune on the current 300-example semantic set.

## Evidence Selection

Evidence is selected in this order:

1. Use `evidence_sentence` when it is non-empty.
2. Otherwise use `text` as a segment-level fallback.
3. If neither exists, mark the evidence as missing.

Each output must record the evidence quality:

```json
{
  "evidence_text": "Leviathan uses web shells to maintain persistence.",
  "evidence_quality": "sentence"
}
```

For a fallback:

```json
{
  "evidence_text": "...",
  "evidence_quality": "segment_fallback"
}
```

Missing evidence must produce `UNCERTAIN`. A segment fallback must not be treated as equivalent to sentence-local evidence.

## Relation Hypothesis Construction

The candidate direction must be preserved exactly. Use fixed relation-specific templates rather than inserting the raw relation label into an arbitrary sentence.

| Relation | Hypothesis template |
|---|---|
| `uses` | `SOURCE uses TARGET.` |
| `targets` | `SOURCE targets TARGET.` |
| `communicates-with` | `SOURCE communicates with TARGET.` |
| `exfiltrates-to` | `SOURCE exfiltrates data to TARGET.` |
| `located-at` | `SOURCE is located at TARGET.` |
| `attributed-to` | `SOURCE is attributed to TARGET.` |
| `authored-by` | `SOURCE is authored by TARGET.` |
| `delivers` | `SOURCE delivers TARGET.` |
| `drops` | `SOURCE drops TARGET.` |
| `indicates` | `SOURCE indicates TARGET.` |
| `owns` | `SOURCE owns TARGET.` |
| `variant-of` | `SOURCE is a variant of TARGET.` |
| `originates-from` | `SOURCE originates from TARGET.` |
| `downloads` | `SOURCE downloads TARGET.` |
| `impersonates` | `SOURCE impersonates TARGET.` |
| `exploits` | `SOURCE exploits TARGET.` |

For example:

```text
SOURCE: Microsoft
RELATION: uses
TARGET: Google

Hypothesis: Microsoft uses Google.
```

The hypothesis must remain fixed across experiments so that changes in results reflect the verifier rather than changing verbalizations.

## NLI Output

The raw model output must be preserved:

```json
{
  "entailment_score": 0.91,
  "contradiction_score": 0.03,
  "neutral_score": 0.06,
  "nli_prediction": "entailment",
  "score_type": "raw_model_score"
}
```

`semantic_score` must not be described as a calibrated probability unless calibration is performed on held-out development data.

The output should also record the exact NLI model identifier and revision.

## Diagnostic Features

V1 should preserve these auxiliary features:

- NLI entailment, contradiction, and neutral scores
- V6 `status`
- `predicate`
- `predicate_status`
- `source_argument`
- `target_argument`
- V2 direction
- V2 argument structure
- `negated`
- `comparison_context`
- `reporting_context`
- `table_context`
- `uncertainty_context`
- `type_compatible`
- SVM `margin`

The SVM margin is only an auxiliary feature. It must never be used as a semantic truth threshold because high-margin predictions can still be false.

The initial baseline should report both:

1. Raw NLI prediction.
2. Hybrid decision after evidence-quality and diagnostic checks.

## Status Policy

The intended policy is:

```text
High entailment and no contradiction -> SUPPORTED
Low entailment or clear contradiction -> UNSUPPORTED
Ambiguous or insufficient evidence   -> UNCERTAIN
```

The following cases should normally abstain or reject rather than be accepted:

- missing evidence
- table or caption context
- negated relation
- comparison or similarity context
- observer/reporting context
- explicit reverse direction
- no detected predicate
- pure entity co-occurrence
- relation not actually expressed in the evidence

These diagnostics should explain and constrain the decision. They should not become another large hard-coded validator.

## Model Strategy

### Phase 1: Zero-Shot Baseline

Run a reproducible pretrained NLI transformer on:

```text
evidence_text -> relation_hypothesis
```

A DeBERTa-v3-base MNLI-style model or comparable pretrained NLI model is appropriate for the first experiment. A Tesla T4 should support inference and later moderate fine-tuning with short sequences.

### Phase 2: Weakly Supervised Development

Potential weak training sources include:

- AZERG labeled relations
- high-quality relation examples with explicit evidence
- synthetic hard negatives

Weak and synthetic labels must be explicitly marked and must not be described as human gold labels.

### Phase 3: Optional Fine-Tuning

Only after inspecting the zero-shot baseline should a trainable verifier be considered. The 300-example semantic set must not be used simultaneously for training and final evaluation.

## Hard Negative Design

The verifier should eventually include difficult negatives such as:

1. Reversed source and target.
2. Wrong subject.
3. Wrong object.
4. Same-sentence co-occurrence without a relation.
5. Coordination cross-pairs.
6. Reporting or observer entity substituted as source.
7. Passive constructions with reversed arguments.
8. Negated relations.
9. Comparison statements.
10. Tables, headings, and captions.
11. Nested entity confusion.
12. Semantically related but incorrect relations.

These are more valuable than only generating unrelated random negatives.

## Data Leakage Prevention

The current annotation status is:

- Examples 1-200: awaiting independent human labels.
- Examples 201-300: provisional assistant-assisted labels.

Therefore the 300-example set is not currently a finalized independent gold standard.

Use explicit provenance fields:

```text
label_source = independent_human
label_source = provisional
label_source = weak
label_source = synthetic
```

Recommended separation:

```text
AZERG / weak / synthetic examples
    -> optional training and development

Independent human examples
    -> frozen evaluation

Provisional examples
    -> development only until independently reannotated
```

Once the independent labels are available, freeze the evaluation file before tuning thresholds or changing the verifier.

## Evaluation Protocol

Compare:

1. SVM only.
2. SVM + V6.
3. SVM + zero-shot NLI.
4. SVM + hybrid verifier.
5. Fine-tuned semantic verifier, if implemented.

Report:

- accuracy
- macro-F1
- per-class precision, recall, and F1
- confusion matrix
- accepted-relation precision
- coverage
- selective accuracy
- uncertainty rate
- relation-wise precision
- error-category distribution

The primary metric for final KG construction is:

```text
precision of accepted SUPPORTED relations
```

Do not claim statistical significance from a small evaluation sample without an appropriate analysis.

## Proposed Output Contract

```json
{
  "source": "Leviathan",
  "source_type": "GROUP",
  "relation": "uses",
  "target": "web shells",
  "target_type": "TECHNIQUE",
  "document": "example-document",
  "segment": 103,
  "evidence_text": "Leviathan uses web shells to maintain persistence.",
  "evidence_quality": "sentence",
  "hypothesis": "Leviathan uses web shells.",
  "nli_prediction": "entailment",
  "entailment_score": 0.91,
  "contradiction_score": 0.03,
  "neutral_score": 0.06,
  "status": "SUPPORTED",
  "semantic_score": 0.91,
  "score_type": "raw_model_score",
  "v6_status": "PLAUSIBLE",
  "diagnostics": {
    "subject_supported": true,
    "predicate_supported": true,
    "object_supported": true,
    "direction_supported": true,
    "coordination_conflict": false,
    "reporting_context": false,
    "negated": false,
    "comparison_context": false,
    "table_context": false
  },
  "provenance": {
    "source_file": "annoctr_relation_predictions_validated_v6.jsonl",
    "svm_margin": 0.42,
    "label_source": "zero_shot_nli"
  }
}
```

The verifier output must be a new file. Existing V2 and V6 files must remain unchanged.

## Final KG Constraint

The existing 1,109 raw semantic relation edges must not be treated as the final KG.

The final graph may only be built after semantic verification demonstrates sufficient precision. Every accepted edge must retain:

- source and target canonical identifiers
- source and target names and types
- relation
- document
- segment
- evidence text
- verification status
- semantic score
- provenance

The intended final pipeline is:

```text
CTI reports
-> NER
-> entity linking
-> candidate generation
-> relation classification
-> structural validation
-> semantic evidence verification
-> evidence-grounded knowledge graph
-> Graph RAG
```

## Current Next Step

Implement a standalone zero-shot inference script that:

1. Reads `data/annoctr_relation_predictions_validated_v6.jsonl`.
2. Selects sentence evidence or records a segment fallback.
3. Builds a fixed relation hypothesis.
4. Runs the selected NLI model.
5. Preserves raw NLI scores.
6. Preserves all original candidate and provenance fields.
7. Writes a new verifier output file.
8. Does not modify V2, V6, or the current raw semantic KG.

No final evaluation or final KG construction should occur until the independent 1-200 labels are available.