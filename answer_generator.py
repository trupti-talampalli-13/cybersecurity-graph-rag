"""Transparent evidence-grounded response formatter (no LLM is called)."""
class AnswerGenerator:
    def generate_answer(self, query, fused_results):
        graph_items = [x for x in fused_results if x.get("type") == "graph"]
        text_items = [x for x in fused_results if x.get("type") == "text"]
        lines = [f"Question: {query}", "", "Retrieved graph assertions:"]

        if not graph_items:
            lines.append("No direct graph assertion was retrieved for the named entities.")
        for item in graph_items:
            lines.extend([
                f"- Claim: {item.get('claim')}",
                f"  Evidence: {item.get('evidence')}",
                f"  Source/date: {item.get('source')} / {item.get('publication_date')}",
                f"  Trust/confidence: {item.get('trust_level')} / {item.get('confidence')}",
                f"  Verification: {item.get('verification_status')}; human verified: {item.get('human_verified')}",
            ])

        lines.extend(["", "Additional document evidence:"])
        if not text_items:
            lines.append("No document passages were retrieved.")
        for item in text_items:
            lines.append(
                f"- [{item.get('source')}, {item.get('publication_date')}, "
                f"report={item.get('report_id')}, segment={item.get('segment_id')}] "
                f"{(item.get('text') or '').strip()}"
            )

        lines.extend([
            "",
            "Limitations:",
            "This output is a structured evidence report, not an LLM-generated explanation.",
            "Automatically extracted graph assertions may be incorrect or only partially supported by their evidence.",
            "A 'trusted' label does not mean a human verified the assertion. Candidate assertions should not be stated as established facts.",
        ])
        return "\n".join(lines)
