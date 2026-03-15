from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from src.ingestion import DocumentChunk
from src.indexing import HybridIndex

class HybridRetriever:
    def __init__(self, index: HybridIndex, cross_encoder_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.index = index
        self.cross_encoder = CrossEncoder(cross_encoder_model_name)
        
    def combine_results(self, bm25_results: List[Tuple[DocumentChunk, float]], 
                        vector_results: List[Tuple[DocumentChunk, float]], 
                        k: int = 60) -> List[Tuple[DocumentChunk, float]]:
        """Combines BM25 and Vector results using Reciprocal Rank Fusion (RRF)."""
        scores = {}
        
        # BM25 RRF
        for rank, (chunk, _) in enumerate(bm25_results):
            content = chunk.content
            scores[content] = scores.get(content, 0) + 1 / (k + rank)
            
        # Vector RRF
        for rank, (chunk, _) in enumerate(vector_results):
            content = chunk.content
            scores[content] = scores.get(content, 0) + 1 / (k + rank)
            
        # Map back to chunks
        # This assumes content is unique enough or we use source/metadata for uniqueness
        content_to_chunk = {chunk.content: chunk for chunk, _ in bm25_results + vector_results}
        
        # Sort and return
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(content_to_chunk[content], score) for content, score in sorted_results]

    def rerank(self, query: str, candidate_chunks: List[DocumentChunk], top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Reranks candidate chunks using a Cross-Encoder."""
        if not candidate_chunks:
            return []
            
        pairs = [[query, chunk.content] for chunk in candidate_chunks]
        scores = self.cross_encoder.predict(pairs)
        
        # Sort by score
        results = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
        return list(results)[:top_k]

    def retrieve(self, query: str, top_k: int = 5, retrieval_top_k: int = 50) -> List[DocumentChunk]:
        """Full hybrid retrieval pipeline: BM25 + Vector -> RRF -> Rerank."""
        # 1. BM25 Search
        bm25_res = self.index.search_bm25(query, top_k=retrieval_top_k)
        
        # 2. Vector Search
        vector_res = self.index.search_vector(query, top_k=retrieval_top_k)
        
        # 3. Combine with RRF
        combined = self.combine_results(bm25_res, vector_res)
        
        # 4. Rerank top results
        candidate_chunks = [chunk for chunk, _ in combined]
        reranked = self.rerank(query, candidate_chunks, top_k=top_k)
        
        return [chunk for chunk, _ in reranked]

if __name__ == "__main__":
    # Test (requires an index or mock)
    print("HybridRetriever loaded.")
