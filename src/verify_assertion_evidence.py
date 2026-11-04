"""Rule-based check of whether one assertion is supported by its own evidence sentence.

This is separate from the pairwise resolver (resolve_assertions.py, which stays frozen as the baseline): it
looks at a single assertion and its evidence_sentence and does not compare assertions with each other.

It is conservative on purpose. When the evidence does not clearly show the claim it abstains instead of
guessing, and it never turns "not observed" into "not used". Its rules are written independently of the
extractor's trigger lexicon (it does not import extract_relations.py), so it is a second look at the same
sentence, but a rule-based one: it can still be wrong, and its output has not been checked by humans.

Checks, in order:
  1 evidence format    missing text, fragment, table-like, question, long segment
  2 entity identity    both entity mentions appear in the sentence
  3 polarity/modality  non-observation, negation, hedging, hypothetical wording
  4 direction          mention order matches active/passive voice
  5 predicate          a cue for the relation sits between the two mentions
  6 context flags      comparison, reported speech, campaign, time wording

Statuses (about one assertion, not the pairwise labels of classify_pair):
  SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, CONTEXT_AMBIGUOUS, REVIEW_REQUIRED

Run from the cti_graph_rag directory to verify every assertion in data/assertions.jsonl:
    python src/verify_assertion_evidence.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUSES = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "CONTEXT_AMBIGUOUS", "REVIEW_REQUIRED")

NON_OBSERVATION = re.compile(
    r"\b(?:(?:did|do|does|have|has|had|could|can|were|was|are|is)(?:\s+not|n't)\s+(?:yet\s+)?"
    r"(?:been\s+)?(?:observ\w+|see\w*|seen|found|find|detect\w*|identif\w+|confirm\w*|verif\w+|"
    r"notic\w+|record\w*|report\w*)|no\s+(?:evidence|sign|indication|trace|instances?)|"
    r"unable\s+to\s+(?:confirm|identify|find|verify|observe)|not\s+(?:yet\s+)?(?:been\s+)?observed)\b", re.I)
NEGATION = re.compile(
    r"\b(?:never|cannot|without|neither|nor|no\s+longer|not(?!\s+(?:only|just|limited|all)\b)|no(?=\s+\w))\b"
    r"|n't\b", re.I)
HEDGE = re.compile(r"\b(?:may|might|could|possibl\w+|probabl\w+|likely|appears?|seems?|suggest\w*|believed|"
                   r"reportedly|allegedly|potentially|perhaps|presumably)\b", re.I)
HYPOTHETICAL = re.compile(r"\b(?:if|would|should|whether|unless|in\s+case)\b", re.I)
COMPARISON = re.compile(r"\b(?:unlike|than|similar\s+to|same\s+as|compared\s+(?:to|with)|in\s+contrast|whereas|"
                        r"rather\s+than|resembl\w+)\b", re.I)
REPORTED = re.compile(r"\b(?:claim\w*|alleg\w+|according\s+to|stated|denied|announced|said)\b", re.I)
TIME_WORDS = re.compile(r"\b(?:since|until|before|after|formerly|previously|(?:19|20)\d\d)\b", re.I)
TABLE_LIKE = re.compile(r"\||\t|(?:\b[0-9a-f]{32,64}\b.*){2,}", re.I)
PASSIVE = re.compile(r"\b(?:\w+ed|\w+en|used|made|run)\s+(?:by|through|via)\b|\b(?:by|through)\b", re.I)

# Cues for each relation, as regex stems; a cue must sit between the two mentions (or contain "against").
PREDICATE_CUES = {
    "USES": r"us(?:e|es|ed|ing)|leverag\w+|employ\w*|utili[sz]\w+|deliver\w*|distribut\w*|download\w*|drop\w*|"
            r"deploy\w*|load\w*|install\w*|execut\w*|rel(?:y|ies|ied)|abus\w+|perform\w*",
    "TARGETS": r"target\w*|against|attack\w*|focus\w*|aim\w*|compromis\w*|victim\w*|infect\w*|hit|hits|affect\w*",
    "EXPLOITS": r"exploit\w*|abus\w+|leverag\w+|take\w*\s+advantage",
    "USES_TECHNIQUE": r"us(?:e|es|ed|ing)|employ\w*|leverag\w+|perform\w*|utili[sz]\w+|via|through|by",
}


def mention_span(sentence, mention):
    """(start, end) of the first whole-word, case-insensitive occurrence of the mention, or None.

    Mentions come from tokenised text ("U.S ." for "U.S."), so spaces between tokens are optional, trailing
    punctuation is ignored and an unreadable character (U+FFFD) matches any one character.
    """
    tokens = re.findall(r"\w+|[^\w\s]", mention or "")
    while tokens and not re.match(r"\w", tokens[-1]):
        tokens.pop()
    if not tokens:
        return None
    pattern = r"\s*".join("." if t == "�" else re.escape(t) for t in tokens)
    m = re.search(r"(?<!\w)" + pattern + r"(?!\w)", sentence, re.I)
    return (m.start(), m.end()) if m else None


def evidence_format(sentence):
    if not sentence or not sentence.strip():
        return "missing"
    words = sentence.split()
    if TABLE_LIKE.search(sentence):
        return "table_like"
    if len(words) < 5:
        return "fragment"
    if sentence.rstrip().endswith("?"):
        return "question"
    if len(sentence) > 600:
        return "long_segment"
    return "sentence"


def verify_assertion(assertion):
    """Return the verification record for one assertion (see the module docstring)."""
    sentence = assertion.get("evidence_sentence") or ""
    result = {"assertion_id": assertion.get("assertion_id"), "status": None, "reason_codes": [],
              "evidence_format": evidence_format(sentence), "identity_check": "not_run",
              "predicate_check": "not_run", "direction_check": "not_run", "polarity_check": "not_run",
              "context_flags": [], "human_review_required": True}

    def finish(status, *codes):
        result["status"] = status
        result["reason_codes"] = list(codes)
        result["human_review_required"] = status != "SUPPORTED"
        return result

    fmt = result["evidence_format"]
    if fmt == "missing":
        return finish("REVIEW_REQUIRED", "MISSING_EVIDENCE")

    subj = mention_span(sentence, assertion.get("subject_mention") or assertion.get("subject"))
    obj = mention_span(sentence, assertion.get("object_mention") or assertion.get("object"))
    if not subj or not obj:
        result["identity_check"] = "mention_missing"
        codes = (["SUBJECT_NOT_IN_EVIDENCE"] if not subj else []) + (["OBJECT_NOT_IN_EVIDENCE"] if not obj else [])
        return finish("INSUFFICIENT_EVIDENCE", *codes)
    if subj[0] < obj[1] and obj[0] < subj[1]:      # the two mentions overlap, so roles cannot be told apart
        result["identity_check"] = "mentions_overlap"
        return finish("INSUFFICIENT_EVIDENCE", "MENTIONS_OVERLAP")
    result["identity_check"] = "both_mentions_present"

    if fmt == "table_like":
        return finish("INSUFFICIENT_EVIDENCE", "TABLE_CO_OCCURRENCE")
    if fmt in ("fragment", "question"):
        return finish("INSUFFICIENT_EVIDENCE", "EVIDENCE_NOT_A_STATEMENT")

    # context flags (informational unless noted below)
    flags = result["context_flags"]
    if COMPARISON.search(sentence):
        flags.append("COMPARISON")
    if REPORTED.search(sentence):
        flags.append("REPORTED_SPEECH")
    if (assertion.get("scope") or {}).get("campaign"):
        flags.append("CAMPAIGN_SCOPED")
    if TIME_WORDS.search(sentence):
        flags.append("TIME_WORDING")
    if fmt == "long_segment":
        flags.append("LONG_SEGMENT")

    # polarity and modality
    if NON_OBSERVATION.search(sentence):
        result["polarity_check"] = "non_observation"
        return finish("REVIEW_REQUIRED", "NON_OBSERVATION", "LIMITED_OBSERVATION_SCOPE")
    lo, hi = sorted([subj, obj])
    # negation only counts near the claim (shortly before the first mention up to the second), so an unrelated
    # "not" later in a long sentence does not flip it
    negated = bool(NEGATION.search(sentence[max(0, lo[0] - 40):hi[1]]))
    claimed_negative = assertion.get("polarity") == "negative"
    if negated and not claimed_negative:
        result["polarity_check"] = "evidence_negated_assertion_positive"
        return finish("CONTRADICTED", "NEGATED_IN_EVIDENCE")
    if claimed_negative and not negated:
        result["polarity_check"] = "assertion_negative_evidence_positive"
        return finish("CONTRADICTED", "ASSERTION_NEGATIVE_BUT_EVIDENCE_POSITIVE")
    result["polarity_check"] = "explicit_negation" if negated else "positive"
    if HYPOTHETICAL.search(sentence):
        return finish("INSUFFICIENT_EVIDENCE", "HYPOTHETICAL_WORDING")

    # direction: active voice puts the subject first, passive voice puts the object first
    first_is_subject = subj[0] < obj[0]
    between = sentence[lo[1]:hi[0]]
    claimed_passive = assertion.get("direction") == "passive"
    if claimed_passive and first_is_subject:
        result["direction_check"] = "mismatch"
        return finish("INSUFFICIENT_EVIDENCE", "DIRECTION_MISMATCH")
    if not claimed_passive and not first_is_subject:
        result["direction_check"] = "mismatch"
        return finish("INSUFFICIENT_EVIDENCE", "DIRECTION_MISMATCH")
    if claimed_passive and not PASSIVE.search(between):
        result["direction_check"] = "no_passive_marker"
        return finish("INSUFFICIENT_EVIDENCE", "PASSIVE_MARKER_MISSING")
    result["direction_check"] = "consistent"

    # predicate cue between the mentions
    if assertion.get("trigger") == "<implicit>":
        result["predicate_check"] = "implicit_trigger"
        return finish("CONTEXT_AMBIGUOUS", "IMPLICIT_PREDICATE")
    cue = PREDICATE_CUES.get(assertion.get("predicate"))
    if not cue or not re.search(r"\b(?:" + cue + r")\b", between, re.I):
        result["predicate_check"] = "no_cue_between_mentions"
        return finish("INSUFFICIENT_EVIDENCE", "NO_PREDICATE_CUE_BETWEEN_MENTIONS")
    result["predicate_check"] = "cue_between_mentions"

    # modality and context that make a supported-looking sentence unsafe to rely on
    if HEDGE.search(sentence):
        return finish("CONTEXT_AMBIGUOUS", "HEDGED_EVIDENCE")
    if "COMPARISON" in flags:
        return finish("CONTEXT_AMBIGUOUS", "COMPARISON_CONTEXT")
    if "REPORTED_SPEECH" in flags:
        return finish("CONTEXT_AMBIGUOUS", "REPORTED_CLAIM")
    return finish("SUPPORTED", "EXPLICIT_STATEMENT" if not negated else "EXPLICIT_NEGATIVE_STATEMENT")


def main():
    path = ROOT / "data" / "assertions.jsonl"
    out = ROOT / "data" / "assertion_verification.jsonl"
    with open(path, encoding="utf-8") as f:
        assertions = [json.loads(line) for line in f]
    results = [verify_assertion(a) for a in assertions]
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Verified {len(results)} assertions -> data/assertion_verification.jsonl")
    for status, n in sorted(Counter(r["status"] for r in results).items()):
        print(f"  {status:22s} {n}")
    print("Reason codes:")
    for code, n in Counter(c for r in results for c in r["reason_codes"]).most_common():
        print(f"  {code:40s} {n}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(__doc__)
    main()
