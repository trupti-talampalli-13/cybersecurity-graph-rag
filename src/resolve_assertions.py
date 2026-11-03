"""Written decision criteria for how two assertions relate, as a small rule-based classifier.

This is a transparent baseline for the contradiction-resolution stage, not a finished resolver. It looks only
at the structure of two assertions (claim, polarity, certainty, dates, campaign); it does not read the evidence
sentences, so it cannot catch cases that depend on wording.

Labels (the project plan's categories, plus "unrelated"):
  supports               same claim, same polarity, compatible scope
  contradicts            opposite polarity, same claim, scope compatible, report/event periods overlap or are close
  temporal_update        opposite polarity, same claim, periods far apart: the claim changed over time
  scope_variant          same claim but different named campaigns: different contexts, not a conflict
  complements            same subject and predicate, both positive, different objects: compatible extra information
  insufficient_evidence  opposite polarity but a hedged statement, or periods neither close nor far apart
  unrelated              different subject or predicate

Period of an assertion: its event span (`event_start`..`event_end`) when it has one, else its publication date.
Thresholds: periods within NEAR_DAYS of each other (or overlapping) are the same period; more than FAR_DAYS apart
are different periods; in between is not decided.

Run from the cti_graph_rag directory to classify two assertion ids:
    python src/resolve_assertions.py A00137 A00152
"""
import datetime
import json
import sys
from pathlib import Path

NEAR_DAYS = 90
FAR_DAYS = 365
LABELS = ("supports", "contradicts", "temporal_update", "scope_variant", "complements",
          "insufficient_evidence", "unrelated")


def period(a):
    """(start, end) as dates: the event span if both ends are known, else the publication date."""
    start, end = a.get("event_start"), a.get("event_end")
    if start and end:
        return datetime.date.fromisoformat(start), datetime.date.fromisoformat(end)
    day = datetime.date.fromisoformat(a["publication_date"])
    return day, day


def gap_days(a, b):
    """Days between the two periods; 0 when they overlap."""
    (s1, e1), (s2, e2) = sorted((period(a), period(b)))
    return max(0, (s2 - e1).days)


def campaign(a):
    """The named campaign of an assertion record (full record with `scope`, or a compact case view)."""
    return a["scope"].get("campaign") if "scope" in a else a.get("campaign")


def campaigns_differ(a, b):
    ca, cb = campaign(a), campaign(b)
    return bool(ca and cb and ca.strip().lower() != cb.strip().lower())


def classify_pair(a, b):
    """Return one of LABELS for two assertion records."""
    if a["subject_id"] != b["subject_id"] or a["predicate"] != b["predicate"]:
        return "unrelated"
    if a["object_id"] != b["object_id"]:
        both_positive = a["polarity"] == b["polarity"] == "positive"
        return "complements" if both_positive else "unrelated"
    # same claim from here on
    if a["polarity"] == b["polarity"]:
        return "scope_variant" if campaigns_differ(a, b) else "supports"
    if a["certainty"] == "hedged" or b["certainty"] == "hedged":
        return "insufficient_evidence"
    if campaigns_differ(a, b):
        return "scope_variant"
    gap = gap_days(a, b)
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
    for line in open(path, encoding="utf-8"):
        record = json.loads(line)
        by_id[record["assertion_id"]] = record
    a, b = by_id[sys.argv[1]], by_id[sys.argv[2]]
    print(f"{a['claim_text']}  vs  {b['claim_text']}")
    print(f"periods {period(a)} and {period(b)}, gap {gap_days(a, b)} days")
    print("->", classify_pair(a, b))


if __name__ == "__main__":
    main()
