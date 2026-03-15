# 📖 Guide 1: Operational Manual (How to Run the System)

This guide covers the setup and execution of the RAG system on your local machine.

## 1. Prerequisites
Ensure you have the following installed:
- **Python 3.10+**
- **NVIDIA Drivers** (for your RTX 3050)
- **Git** (optional, for version control)

## 2. Initial Setup
Open your terminal (PowerShell or Command Prompt) and navigate to the project folder:
```powershell
cd D:\GEMINI\RAG
pip install -r requirements.txt
```

## 3. Running the System
You have two primary ways to interact with the system:

### A. The Web UI (Recommended)
This provides the most interactive and visual experience.
```powershell
streamlit run app.py
```
*A browser window will automatically open at `http://localhost:8501`.*

### B. The CLI (Command Line Interface)
Best for quick queries or automated tasks.
```powershell
# To index new documents and ask a question:
python main.py --index --offline --query "What is attention?"

# To just ask a question using an existing index:
python main.py --offline --query "Summarize the sample document."
```

## 4. Managing Documents
- Place your PDFs, Word docs, Excel files, or Markdown files in the `D:\GEMINI\RAG\documents` folder.
- In the Web UI, you can also use the **File Uploader** to add documents directly.
- **Crucial:** After adding or removing files, always click **"Rebuild Index"** in the UI to update the system's knowledge.
