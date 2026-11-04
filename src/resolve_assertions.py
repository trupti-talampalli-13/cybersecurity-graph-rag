"""Written decision criteria for how two assertions relate, as a small rule-based classifier.

This is a transparent baseline for the contradiction-resolution stage, not a finished resolver. It looks only
at the structure of two assertions (claim, polarity, certainty, dates, campaign); it does not read the evidence
sentences, so it cannot catch cases that depend on wording. The rules are hypotheses, not validated findings:
the tests show the code follows them, not that they resolve real cybersecurity contradictions correctly.

Labels (the project plan's categories, plus "unrelated"):
  supports               same claim, same polarity, compatible scope
  contradicts            opposite polarity, same claim, same scope, periods overlap or are close
  temporal_update        opposite polarity, same claim, same scope, periods far apart: the claim changed over time
  scope_variant          same claim but different named campaigns: different contexts, not a conflict
  complements            same subject and predicate, both positive, different objects: compatible extra information
  insufficient_evidence  opposite polarity but the evidence cannot settle it (see below)
  unrelated              different subject or predicate

Opposite polarity is `insufficient_evidence` when:
  - either assertion is hedged;
  - exactly one of them names a campaign (a general denial and a campaign-specific use are not necessarily in
    conflict, so a scoped claim and an unscoped one are not compared);
  - one period is an event span and the other only a publication date, and they do not overlap (the
    publication-dated assertion does not say when the activity happened);
  - the periods are neither close nor far apart.

Scope: two campaign names are the same after lower-casing and dropping words such as "operation" and "campaign".
No named campaign on either side counts as the same scope; a name on one side only is a mixed scope.

Period of an assertion: its event span (`event_start`..`event_end`) when it has one ("event" basis), else its
publication date ("publication" basis). Periods within NEAR_DAYS of each other (or overlapping) are the same
period; more than FAR_DAYS apart are different periods. The 90 and 365 day thresholds are round numbers chosen
as a quarter and a year, not derived from data; treat them as parameters to be tuned on labelled real pairs.

Run from the cti_graph_rag directory to classify two assertion ids:
    python src/resolve_assertions.py A00137 A00152
"""
import datetime
import json
import re
import sys
from pathlib import Path

NEAR_DAYS = 90
FAR_DAYS = 365
LABELS = ("supports", "contradicts", "temporal_update", "scope_variant", "complements",
          "insufficient_evidence", "unrelated")
CAMPAIGN_FILLER = re.compile(r"\b(?:operation|campaign|the)\b")


def period(a):
    """(start, end) as dates: the event span if both ends are known, else the publication date."""
    start, end = a.get("event_start"), a.get("event_end")
    if start and end:
        return datetime.date.fromisoformat(start), datetime.date.fromisoformat(end)
    day = datetime.date.fromisoformat(a["publication_date"])
    return day, day


def time_basis(a):
    """"event" when the period is a real event span from the sentence, "publication" when it is only the report date."""
    return "event" if a.get("event_start") and a.get("event_end") else "publication"


def gap_days(a, b):
    """Days between the two periods; 0 when they overlap."""
    (s1, e1), (s2, e2) = sorted((period(a), period(b)))
    return max(0, (s2 - e1).days)


def campaign(a):
    """The named campaign of an assertion record (full record with `scope`, or a compact case view)."""
    return a["scope"].get("campaign") if "scope" in a else a.get("campaign")


def normalize_campaign(name):
    """Lower-case, drop filler words ("Operation Alpha" and "Alpha campaign" are the same) and extra spaces."""
    return re.sub(r"\s+", " ", CAMPAIGN_FILLER.sub(" ", name.lower())).strip(" -.,")


def scope_relation(a, b):
    """"same" (no campaign on either side, or equal names), "different" (two different names) or "mixed" (one name)."""
    ca, cb = campaign(a), campaign(b)
    if not ca and not cb:
        return "same"
    if not ca or not cb:
        return "mixed"
    return "same" if normalize_campaign(ca) == normalize_campaign(cb) else "different"


def campaigns_differ(a, b):
    return scope_relation(a, b) == "different"


def classify_pair(a, b):
    """Return one of LABELS for two assertion records."""
    if a["subject_id"] != b["subject_id"] or a["predicate"] != b["predicate"]:
        return "unrelated"
    if a["object_id"] != b["object_id"]:
        both_positive = a["polarity"] == b["polarity"] == "positive"
        return "complements" if both_positive else "unrelated"
    # same claim from here on
    scope = scope_relation(a, b)
    if a["polarity"] == b["polarity"]:
        return "scope_variant" if scope == "different" else "supports"
    # opposite polarity
    if a["certainty"] == "hedged" or b["certainty"] == "hedged":
        return "insufficient_evidence"
    if scope == "different":
        return "scope_variant"
    if scope == "mixed":
        return "insufficient_evidence"
    gap = gap_days(a, b)
    if time_basis(a) != time_basis(b):      # an event span against a bare publication date
        return "contradicts" if gap == 0 else "insufficient_evidence"
    if gap <= NEAR_DAYS:
        return "contradicts"
    if gap >= FAR_DAYS:
        return "temporal_update"
    return "insufficient_evidence"


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    path = Path(__file__).resolve().parent.parent / "data" / "assertions.jsonl"
    by_id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            by_id[record["assertion_id"]] = record
    a, b = by_id[sys.argv[1]], by_id[sys.argv[2]]
    print(f"{a['claim_text']}  vs  {b['claim_text']}")
    print(f"periods {period(a)} ({time_basis(a)}) and {period(b)} ({time_basis(b)}), gap {gap_days(a, b)} days, "
          f"scope {scope_relation(a, b)}")
    print("->", classify_pair(a, b))


if __name__ == "__main__":
    main()
