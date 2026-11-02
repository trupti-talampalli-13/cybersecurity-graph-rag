# Semantic Verifier V2

`src/semantic_verifier_v2.py` implements a conservative, deterministic rule
baseline for checking whether a candidate relation is explicitly expressed by
its evidence. It does not call an NLI model or interpret NLI scores as
probabilities. V1 and V6 scripts and outputs remain unchanged.

## Run

From the repository root:

```powershell
python src/semantic_verifier_v2.py
```

The default input is `data/semantic_verifier_v1_hybrid.jsonl`; the default
output is `data/semantic_verifier_v2.jsonl`. Override either path with
`--input` or `--output`. The script refuses to use the same resolved path for
input and output.

## Decision policy

- `SUPPORTED`: both entity mentions are grounded and an explicitly configured
  relation cue connects them in the candidate direction, with no detected
  blocking context.
- `CONTRADICTED`: an explicit predicate connects the entities in the reverse
  direction, or the candidate-directed predicate is locally negated.
- `INSUFFICIENT_EVIDENCE`: evidence or entity grounding is missing, or no
  configured relation predicate connects the entities.
- `CONTEXT_AMBIGUOUS`: a predicate is present but reporting, scope,
  hypothetical, comparison, coordination, temporal, or negation context
  prevents a safe decision.
- `REVIEW_REQUIRED`: table/list, heading, or caption evidence, or another
  unresolved rule case, requires human inspection.

The explicit relation-cue inventory is versioned in the script as
`semantic_verifier_v2_rules_0.1.0`. It is deliberately finite; unlisted
synonyms and aliases are not inferred. Passive constructions containing `by`
are direction-adjusted for the configured active-verb relations. Identity is
checked by exact normalized surface form only; aliases are not resolved.

V2 adds structured `v2_*` fields without replacing existing input fields.
`v2_nli_scores` copies any available baseline scores and metadata; those scores
cannot override missing entities, missing predicates, or reversed direction.
`v2_legacy_status` maps `SUPPORTED` to `SUPPORTED`, `CONTRADICTED` to
`UNSUPPORTED`, and all abstention/review statuses to `UNVERIFIED`. This
compatibility field is not a claim that insufficient evidence is false.

Only `SUPPORTED` is marked as not requiring human review. Non-supported
decisions remain reviewable and should be retained with their evidence and
provenance rather than silently discarded from an audit workflow.

## Evaluation boundary

The 300-record pilot is exploratory only. Do not tune V2 against it and report
those same records as independent evaluation. Final performance claims require
a separate, document-level held-out set with independent human review. The
baseline fingerprints and script provenance captured before V2 changes are in
[`semantic_verifier_v2_baseline_manifest.json`](./semantic_verifier_v2_baseline_manifest.json).

The initial tests are focused synthetic rule checks, not a substitute for that
human-reviewed evaluation.
