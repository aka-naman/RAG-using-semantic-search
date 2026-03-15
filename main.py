import os
import argparse
from dotenv import load_dotenv
from src.ingestion import load_documents, chunk_documents
from src.indexing import HybridIndex
from src.retrieval import HybridRetriever
from src.generation import OpenAIRAGGenerator, LocalRAGGenerator

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Ask My Docs - Technical RAG System")
    parser.add_argument("--index", action="store_true", help="Force re-indexing of documents")
    parser.add_argument("--query", type=str, help="Question to ask the system")
    parser.add_argument("--offline", action="store_true", help="Use local offline model for generation")
    parser.add_argument("--model", type=str, default=None, help="Specific model ID to use (OpenAI or HF)")
    args = parser.parse_args()

    doc_dir = os.getenv("DOCUMENT_DIR", "./documents")
    index_dir = os.getenv("INDEX_DIR", "./index")
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # 1. Indexing
    if args.index or not os.path.exists(os.path.join(index_dir, "bm25_index.pkl")):
        print("Indexing documents...")
        docs = load_documents(doc_dir)
        if not docs:
            print(f"No documents found in {doc_dir}. Please add some files.")
            return
            
        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"Created {len(chunks)} chunks.")
        
        index = HybridIndex(embedding_model_name=embedding_model)
        index.create_bm25_index(chunks)
        index.create_vector_index(chunks)
        index.save_index(index_dir)
        print(f"Index saved to {index_dir}.")
    else:
        print(f"Loading existing index from {index_dir}...")
        index = HybridIndex.load_index(index_dir, embedding_model_name=embedding_model)

    # 2. Retriever
    retriever = HybridRetriever(index)
    
    # 3. Generator Selection
    if args.offline:
        model_id = args.model if args.model else "unsloth/Llama-3.2-1B-Instruct"
        generator = LocalRAGGenerator(model_id=model_id)
    else:
        model_id = args.model if args.model else os.getenv("MODEL_NAME", "gpt-4o")
        generator = OpenAIRAGGenerator(model_name=model_id)
    
    # 4. Process Query
    if args.query:
        query = args.query
        process_query(query, retriever, generator)
    else:
        # Interactive loop
        while True:
            query = input("\nAsk a question (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            process_query(query, retriever, generator)

def process_query(query, retriever, generator):
    print("\nRetrieving context...")
    context = retriever.retrieve(query)
    
    if not context:
        print("No relevant context found.")
        return

    print("Generating answer...")
    answer = generator.generate_answer(query, context)
    
    print("\n" + "="*50)
    print("ANSWER:")
    print(answer)
    print("="*50)
    
    # Self-correction (currently more effective with OpenAI)
    print("\nSelf-correcting...")
    evaluation = generator.self_correct(query, context, answer)
    print("\nEVALUATION:")
    print(evaluation["evaluation"])
    print("="*50)

if __name__ == "__main__":
    main()
