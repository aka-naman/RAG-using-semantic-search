# Domain-Specific RAG System

This project implements a high-performance, domain-specific "Ask My Docs" system featuring:
- **Hybrid Retrieval:** Combines BM25 (keyword-based) and Vector Search (semantic-based).
- **Cross-Encoder Reranking:** Uses a cross-encoder model to refine retrieval results for higher precision.
- **Self-Correction:** Implements an LLM-based feedback loop to validate retrieval quality and answer completeness.
- **Technical Focus:** Optimized for complex technical queries and specialized domain knowledge.
- **Interactive UI:** Built-in Streamlit app for real-time document chat.
- **Local & Cloud Models:** Support for both OpenAI (GPT-4) and local offline execution (Llama 3.2 via unsloth).

## 🏗️ Architecture

1. **Ingestion Pipeline:** Processes technical documents (PDF, Markdown, etc.) with configurable chunking strategies.
2. **Indexing:**
   - Vector Store for semantic search.
   - BM25 index for precise keyword matching.
3. **Hybrid Retriever:** Merges results from both indices using Reciprocal Rank Fusion (RRF).
4. **Generator:** LLM-powered response generation with built-in hallucination checks and self-correction logic.

## 🚀 Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd RAG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   Copy `.env.template` to `.env` and configure your API keys (e.g., `OPENAI_API_KEY`) if using online mode.

## 💻 Usage

### Streamlit Web Interface (Recommended)
Run the interactive web UI to upload documents, configure retrieval settings, and chat in real-time:
```bash
streamlit run app.py
```

### Command Line Interface
Run queries directly from the terminal or start an interactive prompt:
```bash
python main.py --query "What is the hybrid retrieval mechanism?"
```

#### CLI Options:
- `--index`: Force re-indexing of the `documents/` folder.
- `--offline`: Use the local offline LLM instead of OpenAI.
- `--model <model_id>`: Specify a custom model ID.

## 📁 Directory Structure
- `documents/`: Place your raw PDFs, TXTs, or markdown files here.
- `index/`: Automatically generated vector and BM25 indices (ignored in git).
- `src/`: Core logic for ingestion, indexing, retrieval, and generation.
- `app.py`: Streamlit application entry point.
- `main.py`: CLI entry point.
