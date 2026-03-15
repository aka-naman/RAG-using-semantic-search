# 🧠 Guide 2: Technical Architecture & UI Explained

This guide explains what the various settings do and how the system processes your data.

## 1. UI Settings & What They Mean
In the sidebar of the Web UI, you'll see several sliders and inputs. Here is how to tune them:

### Chunk Size (e.g., 1000)
- **What it is:** The maximum number of characters in each piece of text the system "remembers."
- **Best use:** Use **smaller chunks (400-600)** for very granular facts. Use **larger chunks (1000-1500)** for complex technical explanations where context matters.
- **Example:** If you're indexing a list of product specs, a **size of 500** ensures each product's details stay together. For a legal contract, **1200** might be better to keep whole clauses intact.

### Chunk Overlap (e.g., 200)
- **What it is:** How many characters of "overlap" exist between consecutive chunks.
- **Best use:** Set this to **15-20%** of your chunk size. It ensures that if a sentence is split between two chunks, the context isn't lost.
- **Example:** With a **200-character overlap**, if a sentence about "System Requirements" is split at the end of Chunk A, the beginning of Chunk B will still contain the start of that sentence, providing full context for both.

### Top K Context (e.g., 5)
- **What it is:** The number of most relevant document pieces sent to the LLM to answer your question.
- **Best use:** **3-5** is standard. Increase this if your question requires information spread across many pages, but be careful as it uses more memory/tokens.
- **Example:** If you ask "What are the common themes across all project reports?", you might need a **Top K of 10** to pull info from multiple documents. For a simple "What is the price of X?", a **Top K of 3** is sufficient.

### Rebuild Index
- **What it is:** This triggers the "brain" of the system to re-read your documents, convert them with **Docling**, and create new mathematical representations (embeddings).

## 2. Behind the Scenes: The RAG Architecture
Your system uses a **Hybrid Retrieval** pipeline, which is significantly more accurate than standard search.

1. **Ingestion (Powered by Docling):**
   When you index, **Docling** uses AI models to analyze the document layout. It recognizes tables as tables (not just jumbled text) and converts everything into structured **Markdown**.
2. **The "Two-Brain" Search (Hybrid):**
   - **Brain 1 (BM25):** Acts like a traditional keyword search. It's great for finding specific names or technical terms (e.g., "RTX 3050").
   - **Brain 2 (Vector Search):** Uses **Sentence-Transformers** to understand *meaning*. If you ask about "GPU performance," it knows to look for "graphics cards" even if the word "GPU" isn't there.
3. **Reciprocal Rank Fusion (RRF):**
   The system merges the results from both searches using a mathematical formula to find the "best of both worlds."
4. **Cross-Encoder Reranking:**
   The top candidates are passed through a final, high-precision model that compares your question directly against each chunk to ensure the highest relevancy.
5. **Generation (Local LLM):**
   The top-K chunks are stuffed into a prompt with your question and sent to **Llama-3.2-1B**. The model is instructed to *only* use the provided context to answer, preventing "hallucinations."

## 3. Best Way to Use the System
For the best results:
1. **Be Specific:** Instead of "Tell me about this," ask "What are the specific hardware requirements mentioned in section 2?".
2. **Use Markdown/Tables:** Since we integrated Docling, you can ask the system to "Compare the data in the two tables provided" – it will "see" the table structure perfectly!
3. **Monitor Context:** Use the **"View Retrieved Context"** expander in the UI. If the context shown doesn't contain the answer, your "Top K" might be too low, or the "Chunk Size" might be too small.
