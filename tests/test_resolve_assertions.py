"""Tests for the decision criteria in src/resolve_assertions.py and the constructed conflict set.

Run from the cti_graph_rag directory:  python -m unittest tests.test_resolve_assertions -v
"""
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from resolve_assertions import (FAR_DAYS, LABELS, NEAR_DAYS, classify_pair, gap_days,  # noqa: E402
                                normalize_campaign, period, scope_relation, time_basis)


def make(obj="B", polarity="positive", date="2021-01-01", certainty="asserted", campaign=None, subject="A",
         predicate="USES", event=None):
    return {"subject_id": subject, "predicate": predicate, "object_id": obj, "polarity": polarity,
            "certainty": certainty, "publication_date": date,
            "event_start": event[0] if event else None, "event_end": event[1] if event else None,
            "scope": {"campaign": campaign}}


def days_later(date, n):
    import datetime
    return (datetime.date.fromisoformat(date) + datetime.timedelta(days=n)).isoformat()


class DecisionCriteria(unittest.TestCase):
    def test_same_claim_same_polarity_supports(self):
        self.assertEqual(classify_pair(make(), make(date="2022-06-01")), "supports")

    def test_opposite_polarity_in_the_same_period_contradicts(self):
        self.assertEqual(classify_pair(make(), make(polarity="negative", date=days_later("2021-01-01", 30))),
                         "contradicts")

    def test_opposite_polarity_long_after_is_a_temporal_update(self):
        self.assertEqual(classify_pair(make(), make(polarity="negative", date=days_later("2021-01-01", 500))),
                         "temporal_update")

    def test_opposite_polarity_in_between_is_not_decided(self):
        self.assertEqual(classify_pair(make(), make(polarity="negative", date=days_later("2021-01-01", 200))),
                         "insufficient_evidence")

    def test_threshold_edges(self):
        base = "2021-01-01"
        self.assertEqual(classify_pair(make(date=base), make(polarity="negative", date=days_later(base, NEAR_DAYS))),
                         "contradicts")
        self.assertEqual(classify_pair(make(date=base), make(polarity="negative", date=days_later(base, FAR_DAYS))),
                         "temporal_update")

    def test_a_hedged_denial_is_insufficient_evidence(self):
        b = make(polarity="negative", certainty="hedged", date=days_later("2021-01-01", 10))
        self.assertEqual(classify_pair(make(), b), "insufficient_evidence")
        self.assertEqual(classify_pair(b, make()), "insufficient_evidence")

    def test_different_named_campaigns_are_a_scope_variant_not_a_conflict(self):
        a, b = make(campaign="Alpha"), make(polarity="negative", campaign="Beta")
        self.assertEqual(classify_pair(a, b), "scope_variant")
        self.assertEqual(classify_pair(make(campaign="Alpha"), make(campaign="Beta")), "scope_variant")

    def test_a_missing_campaign_is_not_a_different_campaign(self):
        self.assertEqual(classify_pair(make(campaign="Alpha"), make()), "supports")
        self.assertEqual(classify_pair(make(campaign="alpha"), make(campaign="ALPHA ")), "supports")

    def test_different_objects_of_the_same_subject_and_predicate_complement(self):
        self.assertEqual(classify_pair(make(obj="B"), make(obj="C")), "complements")

    def test_a_negative_with_a_different_object_is_not_a_complement(self):
        self.assertEqual(classify_pair(make(obj="B"), make(obj="C", polarity="negative")), "unrelated")

    def test_different_subject_or_predicate_is_unrelated(self):
        self.assertEqual(classify_pair(make(), make(subject="Z")), "unrelated")
        self.assertEqual(classify_pair(make(), make(predicate="TARGETS")), "unrelated")

    def test_event_spans_are_compared_with_event_spans(self):
        old_event = make(date="2021-01-01", event=("2015-01-01", "2015-12-31"))
        denial_in_2021 = make(polarity="negative", date="2021-01-10", event=("2021-01-01", "2021-01-31"))
        self.assertEqual(classify_pair(old_event, denial_in_2021), "temporal_update")

    def test_an_event_span_against_a_bare_publication_date_is_not_a_temporal_update(self):
        # B is silent on when it happened: it may be about 2015 too, so nothing can be decided
        old_event = make(date="2021-01-01", event=("2015-01-01", "2015-12-31"))
        silent_denial = make(polarity="negative", date="2021-01-10")
        self.assertEqual(classify_pair(old_event, silent_denial), "insufficient_evidence")

    def test_a_mixed_time_basis_is_a_contradiction_only_when_the_periods_overlap(self):
        span = make(event=("2020-01-01", "2020-12-31"))
        inside = make(polarity="negative", date="2020-06-01")
        self.assertEqual(classify_pair(span, inside), "contradicts")

    def test_time_basis(self):
        self.assertEqual(time_basis(make(event=("2020-01-01", "2020-12-31"))), "event")
        self.assertEqual(time_basis(make()), "publication")

    def test_the_plans_example_scoped_use_against_an_unscoped_denial(self):
        report_a = make(date="2023-06-01")                                   # X uses Y
        report_b = make(polarity="negative", date="2025-06-01")              # X does not use Y
        report_c = make(date="2025-06-15", campaign="Campaign B")            # X uses Y in Campaign B
        self.assertEqual(classify_pair(report_a, report_b), "temporal_update")
        self.assertEqual(classify_pair(report_a, report_c), "supports")
        self.assertEqual(classify_pair(report_b, report_c), "insufficient_evidence")

    def test_a_mixed_scope_with_opposite_polarity_is_not_compared(self):
        self.assertEqual(classify_pair(make(campaign="Alpha"), make(polarity="negative")), "insufficient_evidence")
        self.assertEqual(classify_pair(make(), make(polarity="negative", campaign="Alpha")), "insufficient_evidence")

    def test_a_mixed_scope_with_the_same_polarity_still_supports(self):
        self.assertEqual(classify_pair(make(campaign="Alpha"), make()), "supports")

    def test_campaign_names_are_compared_without_filler_words(self):
        self.assertEqual(normalize_campaign("Operation  Alpha"), "alpha")
        self.assertEqual(normalize_campaign("The Alpha Campaign"), "alpha")
        self.assertEqual(scope_relation(make(campaign="Operation Alpha"), make(campaign="alpha campaign")), "same")
        self.assertEqual(scope_relation(make(campaign="Alpha"), make(campaign="Beta")), "different")
        self.assertEqual(scope_relation(make(), make()), "same")
        self.assertEqual(scope_relation(make(campaign="Alpha"), make()), "mixed")

    def test_overlapping_periods_have_no_gap(self):
        a = make(event=("2020-01-01", "2020-12-31"))
        b = make(date="2020-06-01")
        self.assertEqual(gap_days(a, b), 0)

    def test_period_falls_back_to_the_publication_date(self):
        start, end = period(make(date="2021-03-04"))
        self.assertEqual((start.isoformat(), end.isoformat()), ("2021-03-04", "2021-03-04"))

    def test_order_does_not_matter(self):
        pairs = [(make(), make(polarity="negative", date=days_later("2021-01-01", 30))),
                 (make(), make(polarity="negative", date=days_later("2021-01-01", 500))),
                 (make(obj="B"), make(obj="C")), (make(campaign="A"), make(campaign="B"))]
        for a, b in pairs:
            self.assertEqual(classify_pair(a, b), classify_pair(b, a))


class ConstructedSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "data" / "constructed_conflicts.jsonl"
        with open(path, encoding="utf-8") as f:
            cls.cases = [json.loads(line) for line in f]

    def test_every_case_agrees_with_the_classifier(self):
        for case in self.cases:
            self.assertEqual(classify_pair(case["assertion_a"], case["assertion_b"]), case["label"], case["case_id"])

    def test_every_label_is_present_with_enough_cases(self):
        counts = Counter(c["label"] for c in self.cases)
        self.assertEqual(set(counts), set(LABELS))
        self.assertTrue(all(n >= 15 for n in counts.values()), counts)

    def test_constructed_assertions_are_marked(self):
        for case in self.cases:
            for key in ("assertion_a", "assertion_b"):
                a = case[key]
                if a["assertion_id"].endswith("-B") or a["source_document"].startswith("constructed-report"):
                    self.assertTrue(a["constructed"], case["case_id"])

    def test_case_ids_are_unique(self):
        ids = [c["case_id"] for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
