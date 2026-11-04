"""Builds an auditable context with provenance labels."""
class ContextBuilder:
    def __init__(self, max_chars_per_item=2500):
        self.max_chars_per_item = max_chars_per_item

    def build_context(self, fused_results):
        sections = []
        for item in fused_results:
            if item.get("type") == "text":
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                sections.append(
                    "[DOCUMENT EVIDENCE]\n"
                    f"Source: {item.get('source')}\n"
                    f"Publication date: {item.get('publication_date')}\n"
                    f"Report ID: {item.get('report_id')}\n"
                    f"Segment ID: {item.get('segment_id')}\n"
                    f"Retrieved by: {', '.join(item.get('retrieval_methods', []))}\n"
                    f"Text: {text[:self.max_chars_per_item]}"
                )
            elif item.get("type") == "graph":
                sections.append(
                    "[GRAPH ASSERTION — AUTOMATICALLY EXTRACTED]\n"
                    f"Assertion ID: {item.get('assertion_id')}\n"
                    f"Claim: {item.get('claim')}\n"
                    f"Evidence sentence: {item.get('evidence')}\n"
                    f"Source: {item.get('source')}\n"
                    f"Publication date: {item.get('publication_date')}\n"
                    f"Trust level: {item.get('trust_level')}\n"
                    f"Confidence: {item.get('confidence')}\n"
                    f"Verification status: {item.get('verification_status')}\n"
                    f"Human verified: {item.get('human_verified')}\n"
                    "Caution: trust level is not human verification; check whether the evidence sentence directly supports the claim."
                )
        return "\n\n".join(sections)
