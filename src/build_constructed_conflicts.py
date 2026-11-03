"""Build a small constructed test set for contradiction resolution.

The 120 gold-tagged reports contain no negative claims and almost no repeated claims, so they cannot test
contradiction resolution. This builds labelled assertion pairs from real assertions by controlled changes:
a second assertion with the opposite polarity, a later date, a different campaign, or a hedge. Two kinds of
pair are real and need no construction: different objects of the same subject and predicate ("complements"),
and the few claims that really are repeated across reports ("supports").

What it is: a set of cases for checking that a resolver applies written criteria (src/resolve_assertions.py)
consistently. What it is not: evidence about how often conflicts occur in real reports, or a check that the
criteria are right; the labels come from the same criteria the classifier uses. Synthetic assertions and
sentences are marked `constructed: true`.

Reads:  data/assertions.jsonl, data/claims.jsonl
Writes: data/constructed_conflicts.jsonl

Run from the cti_graph_rag directory:  python src/build_constructed_conflicts.py
"""
import datetime
import json
import random
from collections import Counter
from pathlib import Path

from resolve_assertions import FAR_DAYS, NEAR_DAYS, classify_pair, period

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 5
PER_TYPE = 15

VERBS = {"USES": ("uses", "use"), "USES_TECHNIQUE": ("uses the technique", "use the technique"),
         "EXPLOITS": ("exploits", "exploit"), "TARGETS": ("targets", "target")}
FIELDS = ("assertion_id", "claim_key", "subject", "subject_id", "predicate", "object", "object_id", "polarity",
          "certainty", "publication_date", "event_start", "event_end", "source_document", "source_vendor",
          "evidence_sentence")


def shift(day, days):
    return (datetime.date.fromisoformat(day) + datetime.timedelta(days=days)).isoformat()


def after(a, days):
    """A date `days` after the end of the assertion's own period (its event span, else its publication date)."""
    return shift(period(a)[1].isoformat(), days)


def sentence(a, polarity, hedged=False, campaign=None):
    """A short synthetic evidence sentence for a constructed assertion."""
    verb, bare = VERBS[a["predicate"]]
    s, o = a["subject_mention"], a["object_mention"]
    where = f" in the {campaign} campaign" if campaign else ""
    if hedged:
        return f"It is unlikely that {s} {verb}s {o}{where}.".replace("uses the techniques", "uses the technique")
    if polarity == "negative":
        return f"{s} does not {bare} {o}{where}."
    return f"{s} {verb} {o}{where}."


def view(a, constructed):
    out = {k: a.get(k) for k in FIELDS}
    out["campaign"] = a["scope"].get("campaign")
    out["constructed"] = constructed
    return out


def variant(a, case, **changes):
    """A constructed second assertion about the same claim."""
    b = dict(a)
    b["scope"] = dict(a["scope"])
    b.update(assertion_id=f"{case}-B", source_document=f"constructed-report-{case}", source_vendor="Constructed",
             event_start=None, event_end=None, certainty="asserted")
    campaign = changes.pop("campaign", a["scope"].get("campaign"))
    b["scope"]["campaign"] = campaign
    b.update(changes)
    b["evidence_sentence"] = sentence(b, b["polarity"], b["certainty"] == "hedged", campaign)
    return b


def main():
    rng = random.Random(SEED)
    assertions = [json.loads(line) for line in open(DATA / "assertions.jsonl", encoding="utf-8")]
    claims = [json.loads(line) for line in open(DATA / "claims.jsonl", encoding="utf-8")]
    by_id = {a["assertion_id"]: a for a in assertions}
    positive = sorted((a for a in assertions if a["polarity"] == "positive" and a["trust_level"] == "trusted"
                       and a["certainty"] == "asserted"), key=lambda a: a["assertion_id"])
    seen, bases = set(), []
    for a in positive:                       # one base per claim
        if a["claim_key"] not in seen:
            seen.add(a["claim_key"])
            bases.append(a)
    cases = []

    def add(label, criterion, a, b, constructed_a, constructed_b):
        if classify_pair(a, b) != label:
            raise SystemExit(f"inconsistent case {label}: classifier says {classify_pair(a, b)}")
        cases.append({"case_id": f"C{len(cases) + 1:03d}", "label": label, "criterion": criterion,
                      "assertion_a": view(a, constructed_a), "assertion_b": view(b, constructed_b)})

    # supports: the claims that really are repeated across reports (real), then constructed repeats
    for c in claims:
        if c["n_reports"] > 1:
            first = {}
            for x in c["assertions"]:
                first.setdefault(x["source_document"], x["assertion_id"])
            ids = sorted(first.values())
            add("supports", "same claim, same polarity, different reports (real)", by_id[ids[0]], by_id[ids[1]],
                False, False)
    for a in rng.sample(bases, PER_TYPE):
        b = variant(a, f"S{len(cases)}", publication_date=shift(a["publication_date"], rng.randint(200, 600)))
        add("supports", "same claim, same polarity, a later report", a, b, False, True)

    for a in rng.sample(bases, PER_TYPE):
        b = variant(a, f"K{len(cases)}", polarity="negative",
                    publication_date=after(a, rng.randint(0, 60)))
        add("contradicts", f"opposite polarity, same claim, periods within {NEAR_DAYS} days", a, b, False, True)

    for a in rng.sample(bases, PER_TYPE):
        b = variant(a, f"T{len(cases)}", polarity="negative",
                    publication_date=after(a, rng.randint(FAR_DAYS + 135, FAR_DAYS + 800)))
        add("temporal_update", f"opposite polarity, same claim, periods at least {FAR_DAYS} days apart", a, b,
            False, True)

    for a in rng.sample(bases, PER_TYPE):
        a2 = dict(a)
        a2["scope"] = {**a["scope"], "campaign": "Operation Alpha"}
        a2["evidence_sentence"] = sentence(a2, a2["polarity"], False, "Operation Alpha")
        b = variant(a, f"V{len(cases)}", polarity=rng.choice(["positive", "negative"]), campaign="Operation Beta",
                    publication_date=a["publication_date"])
        add("scope_variant", "same claim, two different named campaigns", a2, b, True, True)

    for a in rng.sample(bases, PER_TYPE):
        b = variant(a, f"I{len(cases)}", polarity="negative", certainty="hedged",
                    publication_date=after(a, rng.randint(0, 60)))
        add("insufficient_evidence", "opposite polarity, but the denial is hedged", a, b, False, True)

    for a in rng.sample(bases, PER_TYPE):
        b = variant(a, f"G{len(cases)}", polarity="negative",
                    publication_date=after(a, rng.randint(NEAR_DAYS + 60, FAR_DAYS - 60)))
        add("insufficient_evidence", f"opposite polarity, periods {NEAR_DAYS}-{FAR_DAYS} days apart", a, b,
            False, True)

    by_subject = {}
    for a in bases:
        by_subject.setdefault((a["subject_id"], a["predicate"]), []).append(a)
    pairs = [(x, y) for group in by_subject.values() for x, y in zip(group, group[1:])
             if x["object_id"] != y["object_id"]]
    for x, y in rng.sample(pairs, min(PER_TYPE, len(pairs))):
        add("complements", "same subject and predicate, different objects (real)", x, y, False, False)

    others = [(x, y) for x, y in zip(bases, rng.sample(bases, len(bases)))
              if (x["subject_id"], x["predicate"]) != (y["subject_id"], y["predicate"])]
    for x, y in others[:PER_TYPE]:
        add("unrelated", "different subject or predicate (real)", x, y, False, False)

    with open(DATA / "constructed_conflicts.jsonl", "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    counts = Counter(c["label"] for c in cases)
    real = sum(not c["assertion_a"]["constructed"] and not c["assertion_b"]["constructed"] for c in cases)
    print(f"Wrote {len(cases)} cases to data/constructed_conflicts.jsonl: {dict(counts)}; "
          f"{real} use only real assertions, {len(cases) - real} include a constructed one")


if __name__ == "__main__":
    main()
