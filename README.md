# CurrectiveRag

A **Corrective RAG** pipeline that combines query expansion, hybrid retrieval, cross-encoder reranking, context compression, and a hallucination check to produce grounded answers from your documents. The pipeline is orchestrated with **LangGraph** and uses **Qwen** (Hugging Face) as the LLM—no OpenAI/ChatGPT required.

## Pipeline Overview

```
User Query
     ↓
Query Expansion
     ↓
Hybrid Retrieval (BM25 + FAISS)
     ↓
Cross-Encoder Reranking
     ↓
Doc Filtering
     ↓
Context Compression
     ↓
LLM Generation
     ↓
Hallucination Check
     ↓
Final Answer
```

| Step | Description |
|------|-------------|
| **Query Expansion** | Expands the user question into 1–3 search queries for better recall. |
| **Hybrid Retrieval** | Combines BM25 (keyword) and FAISS (semantic) retrieval with configurable weights. |
| **Cross-Encoder Reranking** | Reranks candidates with a cross-encoder for higher precision. |
| **Doc Filtering** | Keeps only documents above a relevance score threshold. |
| **Context Compression** | Sentence-level filtering so only context that helps answer the question is kept. |
| **LLM Generation** | Qwen generates a draft answer from the compressed context only. |
| **Hallucination Check** | Verifies the draft is grounded in the context. |
| **Final Answer** | Returns the draft if grounded, otherwise a safe fallback. |

## Requirements

- Python 3.10+
- Optional: GPU for faster Qwen and cross-encoder inference

## Setup

1. **Clone and enter the project**
   ```bash
   cd CurrectiveRag
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # or:  .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. **Add your documents**  
   Place PDFs in a `documents/` folder. The default code loads:
   - `./documents/Deep+Learning+Ian+Goodfellow.pdf`
   - `./documents/ml_ebook.pdf`  

   Edit the “Load Documents” section in `CRag.py` to point to your own files.

4. **Environment (optional)**  
   Create a `.env` file if you use any env-based config (e.g. for future API keys). The pipeline runs fully local with Qwen and does not require API keys by default.

## Usage

Run the pipeline (builds the index, runs the example question, and prints the final answer):

```bash
python CRag.py
```

The script will:

1. Load and chunk the PDFs  
2. Build FAISS and BM25 indexes and the hybrid retriever  
3. Load the Qwen model and cross-encoder  
4. Run the graph with the example question: *"Batch normalization vs layer normalization"*  
5. Print expanded queries, hallucination check result, and the final answer  

### Using the pipeline in code

After running the script, the compiled app is available as `app`. You can invoke it with a custom question and initial state:

```python
res = app.invoke({
    "question": "Your question here",
    "expanded_queries": [],
    "docs": [],
    "reranked_docs": [],
    "filtered_docs": [],
    "compressed_context": "",
    "draft_answer": "",
    "hallucination_ok": False,
    "final_answer": "",
})

print(res["final_answer"])
print("Hallucination OK:", res["hallucination_ok"])
```

## Project structure

```
CurrectiveRag/
├── CRag.py           # Main pipeline (LangGraph + Qwen)
├── CRag.ipynb        # Notebook version (if used)
├── documents/        # PDFs to index (add your own)
├── requirements.txt
└── README.md
```

## Main dependencies

- **LangChain / LangGraph** – orchestration and retrieval utilities  
- **langchain-huggingface** – Hugging Face embeddings (sentence-transformers)  
- **FAISS** – vector store for semantic search  
- **rank_bm25** – BM25 retriever  
- **langchain-classic** – `EnsembleRetriever` for hybrid search  
- **sentence-transformers** – embeddings and cross-encoder reranker  
- **transformers** – Qwen model (e.g. `Qwen/Qwen2-1.5B-Instruct`)  
- **pypdf** – PDF loading  

## Configuration

You can tune behavior in `CRag.py`:

- **Hybrid weights**: `EnsembleRetriever(..., weights=[0.4, 0.6])` (BM25 vs FAISS)  
- **Rerank size**: `RERANK_TOP_K = 6`  
- **Doc filtering**: `DOC_SCORE_THRESHOLD = 0.3`  
- **Chunking**: `chunk_size=900`, `chunk_overlap=150`  
- **Qwen model**: `model_name = "Qwen/Qwen2-1.5B-Instruct"` (swap for larger Qwen2 if needed)  

## License

Use and modify as needed for your project.
