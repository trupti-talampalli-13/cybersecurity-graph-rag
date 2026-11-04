"""End-to-end Graph-Augmented RAG orchestration."""
from src.hybrid_graph_rag import HybridGraphRAG
from src.evidence_fusion import EvidenceFusion
from src.context_builder import ContextBuilder
from src.answer_generator import AnswerGenerator


class GraphRAGPipeline:
    def __init__(
        self,
        data_path="data/processed_segments.jsonl",
        graph_path="data/cti_knowledge_graph_trusted.json",
        assertions_path="data/assertions.jsonl",
        retriever=None,
    ):
        self.retriever = retriever or HybridGraphRAG(
            data_path=data_path,
            graph_path=graph_path,
            assertions_path=assertions_path,
        )
        self.fusion = EvidenceFusion()
        self.context_builder = ContextBuilder()
        self.answer_generator = AnswerGenerator()

    def ask(self, query, top_k=5, fused_top_k=10):
        all_results = self.retriever.retrieve(query, top_k=top_k)
        fused = self.fusion.fuse(all_results, top_k=fused_top_k)
        context = self.context_builder.build_context(fused)
        answer = self.answer_generator.generate_answer(query, fused)
        return {
            "query": query,
            "answer": answer,
            "context": context,
            "retrieval_results": all_results,
            "fused_results": fused,
        }
