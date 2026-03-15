import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer
from src.ingestion import DocumentChunk

class HybridIndex:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.bm25 = None
        self.vector_index = None
        self.chunks = []
        
    def create_bm25_index(self, chunks: List[DocumentChunk]):
        """Creates a BM25 index from document chunks."""
        tokenized_corpus = [chunk.content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.chunks = chunks

    def create_vector_index(self, chunks: List[DocumentChunk]):
        """Creates a FAISS vector index from document chunks."""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatL2(dimension)
        self.vector_index.add(np.array(embeddings).astype('float32'))
        
    def save_index(self, index_dir: str):
        """Saves both BM25 and Vector indices."""
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
            
        # Save BM25
        with open(os.path.join(index_dir, "bm25_index.pkl"), "wb") as f:
            pickle.dump((self.bm25, self.chunks), f)
            
        # Save Vector Index
        faiss.write_index(self.vector_index, os.path.join(index_dir, "vector_index.bin"))
        
    @classmethod
    def load_index(cls, index_dir: str, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """Loads both BM25 and Vector indices."""
        hybrid_index = cls(embedding_model_name)
        
        # Load BM25
        with open(os.path.join(index_dir, "bm25_index.pkl"), "rb") as f:
            hybrid_index.bm25, hybrid_index.chunks = pickle.load(f)
            
        # Load Vector Index
        hybrid_index.vector_index = faiss.read_index(os.path.join(index_dir, "vector_index.bin"))
        
        return hybrid_index

    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Searches the BM25 index."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:top_k]
        
        return [(self.chunks[i], scores[i]) for i in top_n]

    def search_vector(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Searches the Vector index."""
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.vector_index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                # Convert distance to score (e.g., higher is better)
                score = 1 / (1 + distances[0][i])
                results.append((self.chunks[idx], score))
                
        return results

if __name__ == "__main__":
    # Example usage (test)
    from src.ingestion import DocumentChunk
    
    test_chunks = [
        DocumentChunk(content="Python is a popular programming language for data science.", metadata={"id": 1}),
        DocumentChunk(content="Machine learning is a subset of artificial intelligence.", metadata={"id": 2}),
        DocumentChunk(content="The RAG system uses hybrid retrieval for better results.", metadata={"id": 3})
    ]
    
    index = HybridIndex()
    index.create_bm25_index(test_chunks)
    index.create_vector_index(test_chunks)
    
    print("BM25 Search for 'RAG':")
    for chunk, score in index.search_bm25("RAG", top_k=2):
        print(f"Score: {score:.4f}, Content: {chunk.content}")
        
    print("\nVector Search for 'AI':")
    for chunk, score in index.search_vector("AI", top_k=2):
        print(f"Score: {score:.4f}, Content: {chunk.content}")
