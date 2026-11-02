"""Tests for the trust policy and graph validation in src/build_trusted_graph.py.

Run from the cti_graph_rag directory:  python -m unittest tests.test_trusted_graph -v
"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_trusted_graph import assign_trust, validate  # noqa: E402

SCHEMA = {
    "USES": {"head": ["GROUP", "MALWARE"], "tail": ["MALWARE", "TOOL"]},
    "USES_TECHNIQUE": {"head": ["GROUP", "MALWARE", "TOOL"], "tail": ["TECHNIQUE", "TACTIC"]},
    "EXPLOITS": {"head": ["GROUP", "MALWARE", "TOOL"], "tail": ["CVE"]},
    "TARGETS": {"head": ["GROUP", "MALWARE"], "tail": ["ORG", "SECTOR", "LOC"]},
}


def props(head_type="GROUP", tail_type="MALWARE", confidence="high"):
    return {"head_type": head_type, "tail_type": tail_type, "confidence": confidence}


def trust(relation, head_type, tail_type, confidence="high", head_node=None, tail_node=None):
    return assign_trust(relation, props(head_type, tail_type, confidence),
                        head_node or {"entity_type": head_type}, tail_node or {"entity_type": tail_type}, SCHEMA)


class TrustPolicy(unittest.TestCase):
    def test_actor_headed_uses_is_trusted(self):
        self.assertEqual(trust("USES", "GROUP", "MALWARE"), ("trusted", []))

    def test_actor_headed_targets_is_trusted_at_medium_confidence(self):
        self.assertEqual(trust("TARGETS", "GROUP", "SECTOR", "medium")[0], "trusted")

    def test_technique_relation_is_trusted_even_with_malware_head(self):
        self.assertEqual(trust("USES_TECHNIQUE", "MALWARE", "TACTIC")[0], "trusted")

    def test_malware_headed_uses_needs_verification(self):
        level, reasons = trust("USES", "MALWARE", "MALWARE")
        self.assertEqual(level, "candidate")
        self.assertIn("malware_or_tool_headed", reasons)

    def test_malware_headed_targets_needs_verification(self):
        self.assertEqual(trust("TARGETS", "MALWARE", "LOC")[0], "candidate")

    def test_exploits_is_candidate(self):
        self.assertEqual(trust("EXPLOITS", "MALWARE", "CVE")[0], "candidate")

    def test_low_confidence_is_low_even_for_actor(self):
        self.assertEqual(trust("TARGETS", "GROUP", "LOC", "low")[0], "low")

    def test_linked_node_type_mismatch_downgrades(self):
        level, reasons = trust("USES", "GROUP", "MALWARE", tail_node={"entity_type": "TECHNIQUE"})
        self.assertEqual(level, "candidate")
        self.assertIn("tail_type_mismatch", reasons)


def small_graph():
    edge_props = {"layer": "relation", "document": "d", "segment": 1, "source": "Proofpoint",
                  "publication_date": "2021-05-18", "evidence_text": "TA505 uses Dridex.", "trigger": "uses",
                  "direction": "active", "confidence": "high", "head_mention": "TA505", "tail_mention": "Dridex",
                  "head_type": "GROUP", "tail_type": "MALWARE", "trust_level": "trusted", "human_verified": False}
    return {"nodes": [{"id": "A"}, {"id": "B"}],
            "edges": [{"source": "A", "target": "B", "relation": "USES", "properties": edge_props}]}


class Validation(unittest.TestCase):
    def test_sound_graph_has_no_problems(self):
        self.assertEqual(validate(small_graph()), [])

    def test_dangling_edge_is_reported(self):
        g = small_graph()
        g["edges"][0]["target"] = "Z"
        self.assertTrue(any("dangling" in p for p in validate(g)))

    def test_missing_provenance_is_reported(self):
        g = copy.deepcopy(small_graph())
        g["edges"][0]["properties"]["evidence_text"] = ""
        self.assertTrue(any("missing" in p for p in validate(g)))

    def test_bad_date_is_reported(self):
        g = small_graph()
        g["edges"][0]["properties"]["publication_date"] = "May 18, 2021"
        self.assertTrue(any("bad date" in p for p in validate(g)))

    def test_claiming_human_verification_is_rejected(self):
        g = small_graph()
        g["edges"][0]["properties"]["human_verified"] = True
        self.assertTrue(any("human_verified" in p for p in validate(g)))


if __name__ == "__main__":
    unittest.main()
