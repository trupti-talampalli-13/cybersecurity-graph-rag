import unittest
from src.evaluation import retrieval_metrics


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_and_mrr(self):
        results = [
            {"report_id": "R1", "segment_id": "S1"},
            {"report_id": "R2", "segment_id": "S2"},
            {"report_id": "R3", "segment_id": "S3"},
        ]
        qrels = {"R2::S2", "R3::S3"}
        metrics = retrieval_metrics(results, qrels, k_values=(1, 2, 3))
        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)


if __name__ == "__main__":
    unittest.main()
