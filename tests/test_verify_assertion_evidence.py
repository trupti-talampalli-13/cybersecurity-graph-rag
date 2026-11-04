"""Controlled cases for src/verify_assertion_evidence.py: each behaviour is tested on its own sentence.

These show the rules do what they are written to do on hand-made sentences, not that they verify real reports.
Run from the cti_graph_rag directory:  python -m unittest tests.test_verify_assertion_evidence -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from verify_assertion_evidence import STATUSES, verify_assertion  # noqa: E402


def make(sentence, subject="APT29", obj="Mimikatz", predicate="USES", polarity="positive", direction="active",
         trigger="uses", campaign=None):
    return {"assertion_id": "T1", "subject": subject, "subject_mention": subject, "object": obj,
            "object_mention": obj, "predicate": predicate, "polarity": polarity, "direction": direction,
            "trigger": trigger, "evidence_sentence": sentence, "scope": {"campaign": campaign}}


def status(sentence, **kw):
    return verify_assertion(make(sentence, **kw))["status"]


class Verifier(unittest.TestCase):
    def test_explicit_positive_statement_is_supported(self):
        r = verify_assertion(make("In 2021 APT29 uses Mimikatz to dump credentials from the victim hosts."))
        self.assertEqual(r["status"], "SUPPORTED")
        self.assertFalse(r["human_review_required"])

    def test_explicit_negative_statement_is_supported_as_a_negative(self):
        self.assertEqual(status("The report states that APT29 never uses Mimikatz in this toolset.",
                                polarity="negative"), "SUPPORTED")

    def test_non_observation_needs_review_and_is_not_a_negative_claim(self):
        r = verify_assertion(make("Researchers did not observe APT29 using Mimikatz during the investigation.",
                                  polarity="negative", trigger="using"))
        self.assertEqual(r["status"], "REVIEW_REQUIRED")
        self.assertEqual(r["reason_codes"], ["NON_OBSERVATION", "LIMITED_OBSERVATION_SCOPE"])

    def test_no_evidence_wording_is_non_observation(self):
        self.assertEqual(status("There is no evidence that APT29 used Mimikatz in this intrusion."),
                         "REVIEW_REQUIRED")

    def test_wrong_subject(self):
        self.assertEqual(status("Lazarus uses Mimikatz to dump credentials from the victim hosts."),
                         "INSUFFICIENT_EVIDENCE")

    def test_wrong_object(self):
        self.assertEqual(status("APT29 uses PowerShell to dump credentials from the victim hosts."),
                         "INSUFFICIENT_EVIDENCE")

    def test_reversed_relationship(self):
        self.assertEqual(status("Mimikatz uses APT29 to dump credentials from the victim hosts."),
                         "INSUFFICIENT_EVIDENCE")

    def test_passive_voice_is_supported_when_marked_passive(self):
        self.assertEqual(status("Mimikatz is used by APT29 to dump credentials from the victim hosts.",
                                direction="passive", trigger="used by"), "SUPPORTED")

    def test_passive_claim_without_passive_marker_abstains(self):
        self.assertEqual(status("Mimikatz and APT29 appear in the same incident report today.",
                                direction="passive"), "INSUFFICIENT_EVIDENCE")

    def test_hedged_statement_is_context_ambiguous(self):
        self.assertEqual(status("APT29 may use Mimikatz to dump credentials from the victim hosts."),
                         "CONTEXT_AMBIGUOUS")

    def test_hypothetical_statement_is_insufficient(self):
        self.assertEqual(status("If APT29 uses Mimikatz the defenders would see the credential dumps."),
                         "INSUFFICIENT_EVIDENCE")

    def test_table_co_occurrence_is_insufficient(self):
        self.assertEqual(status("APT29 | Mimikatz | credential theft | Windows"), "INSUFFICIENT_EVIDENCE")

    def test_cooccurrence_without_a_relationship_cue_is_insufficient(self):
        self.assertEqual(status("The APT29 report discusses Mimikatz in the appendix of the paper."),
                         "INSUFFICIENT_EVIDENCE")

    def test_missing_evidence_needs_review(self):
        self.assertEqual(status(""), "REVIEW_REQUIRED")
        self.assertEqual(status(None), "REVIEW_REQUIRED")

    def test_negated_evidence_contradicts_a_positive_assertion(self):
        self.assertEqual(status("The group APT29 does not use Mimikatz to dump credentials any more."),
                         "CONTRADICTED")

    def test_positive_evidence_contradicts_a_negative_assertion(self):
        self.assertEqual(status("The group APT29 uses Mimikatz to dump credentials from the victim hosts.",
                                polarity="negative"), "CONTRADICTED")

    def test_comparison_is_context_ambiguous(self):
        self.assertEqual(status("Unlike Lazarus, APT29 uses Mimikatz to dump credentials from the hosts."),
                         "CONTEXT_AMBIGUOUS")

    def test_reported_claim_is_context_ambiguous(self):
        self.assertEqual(status("A vendor claimed that APT29 uses Mimikatz to dump credentials from hosts."),
                         "CONTEXT_AMBIGUOUS")

    def test_implicit_trigger_is_context_ambiguous(self):
        self.assertEqual(status("APT29 relied on a dropper that ran Mimikatz on the victim hosts later.",
                                trigger="<implicit>"), "CONTEXT_AMBIGUOUS")

    def test_targets_relation(self):
        self.assertEqual(status("The group Winnti targets German chemical companies in the region.",
                                subject="Winnti", obj="German chemical companies", predicate="TARGETS",
                                trigger="targets"), "SUPPORTED")

    def test_every_status_is_a_known_status_and_flags_are_reported(self):
        r = verify_assertion(make("Since 2019 APT29 uses Mimikatz to dump credentials from the hosts.",
                                  campaign="Operation X"))
        self.assertIn(r["status"], STATUSES)
        self.assertIn("CAMPAIGN_SCOPED", r["context_flags"])
        self.assertIn("TIME_WORDING", r["context_flags"])


if __name__ == "__main__":
    unittest.main()
