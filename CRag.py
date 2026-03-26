"""
Corrective RAG pipeline (high-level):

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
"""
from typing import List, TypedDict, Literal
from pydantic import BaseModel
import os
import re
import json

from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
# -----------------------------
# LangChain Imports
# -----------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.retrievers import EnsembleRetriever

# -----------------------------
# LangGraph
# -----------------------------
from langgraph.graph import StateGraph, START, END

# -----------------------------
# Transformers (Qwen) + Cross-Encoder
# -----------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import CrossEncoder

# -----------------------------
# Load Documents
# -----------------------------
docs = (
    PyPDFLoader("./documents/Deep+Learning+Ian+Goodfellow.pdf").load()
    + PyPDFLoader("./documents/ml_ebook.pdf").load()
)

# -----------------------------
# Split into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(docs)

# Clean encoding issues
for d in chunks:
    d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8")

# -----------------------------
# Embeddings (Hugging Face)
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Vector Store (FAISS) + BM25 → Hybrid Retriever
# -----------------------------
vector_store = FAISS.from_documents(chunks, embeddings)
faiss_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 12})
bm25_retriever = BM25Retriever.from_documents(chunks, k=12)
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6],
)

# -----------------------------
# Cross-Encoder for reranking
# -----------------------------
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K = 6

# -----------------------------
# Load Qwen Model
# -----------------------------
model_name = "Qwen/Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"  # auto uses GPU if available
)

# -----------------------------
# Create Pipeline (Qwen text-generation)
# -----------------------------
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,
    do_sample=True,
)

# -----------------------------
# Qwen chat invoker: format with model's chat template, then generate
# -----------------------------
def _messages_to_qwen_prompt(messages: list) -> str:
    """Convert LangChain-style messages to Qwen chat format using tokenizer template."""
    role_map = {"system": "system", "human": "user", "user": "user", "ai": "assistant", "assistant": "assistant"}
    qwen_messages = []
    for m in messages:
        role = getattr(m, "type", None) or (m.get("type") if isinstance(m, dict) else "user")
        content = getattr(m, "content", None) or (m.get("content", "") if isinstance(m, dict) else str(m))
        qwen_messages.append({"role": role_map.get(role, "user"), "content": content})
    prompt = tokenizer.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def _extract_json(text: str) -> str:
    """Extract JSON object or array from model output (handles markdown code blocks)."""
    text = text.strip()
    # Try ```json ... ``` first
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    # Else first {...} or [...]
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        return m.group(1)
    return text


def qwen_invoke(messages: list, *, parse_json: bool = False, output_schema: type = None):
    """Run Qwen with chat-formatted messages. Optionally parse JSON into output_schema."""
    prompt = _messages_to_qwen_prompt(messages)
    out = pipe(prompt)
    generated = out[0]["generated_text"]
    # Generated text includes the prompt; take only the new part (after the last turn)
    if prompt in generated:
        response = generated[len(prompt):].strip()
    else:
        response = generated.strip()
    if parse_json and output_schema:
        raw = _extract_json(response)
        data = json.loads(raw)
        return output_schema.model_validate(data)
    return response

print("✅ Setup completed successfully!")

# -----------------------------
# State (high-level pipeline)
# -----------------------------
class State(TypedDict):
    question: str
    expanded_queries: List[str]
    docs: List[Document]           # after hybrid retrieval
    reranked_docs: List[Document]   # after cross-encoder
    filtered_docs: List[Document]   # after doc filtering
    compressed_context: str
    draft_answer: str
    hallucination_ok: bool
    final_answer: str

# -----------------------------
# 1. Query Expansion
# -----------------------------
query_expansion_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the user question, output 1 to 3 short search queries that help find relevant documents.\n"
            "Output JSON only: {\"queries\": [\"query1\", \"query2\", ...]}",
        ),
        ("human", "Question: {question}"),
    ]
)


class ExpandedQueries(BaseModel):
    queries: List[str]


def _query_expansion_invoke(question: str) -> List[str]:
    messages = query_expansion_prompt.invoke({"question": question}).to_messages()
    out = qwen_invoke(messages, parse_json=True, output_schema=ExpandedQueries)
    queries = out.queries if out.queries else [question]
    return [question] + [q for q in queries if q != question][:2]  # original + up to 2 more


def query_expansion_node(state: State) -> State:
    expanded = _query_expansion_invoke(state["question"])
    return {"expanded_queries": expanded}

# -----------------------------
# 2. Hybrid Retrieval
# -----------------------------
def hybrid_retrieval_node(state: State) -> State:
    queries = state.get("expanded_queries") or [state["question"]]
    seen: set = set()
    merged: List[Document] = []
    for q in queries:
        for doc in hybrid_retriever.invoke(q):
            key = (doc.page_content[:200],)
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return {"docs": merged[:20]}

# -----------------------------
# 3. Cross-Encoder Reranking
# -----------------------------
def cross_encoder_rerank_node(state: State) -> State:
    q = state["question"]
    docs = state["docs"]
    if not docs:
        return {"reranked_docs": []}
    pairs = [(q, d.page_content) for d in docs]
    scores = cross_encoder.predict(pairs)
    indexed = list(zip(scores, docs))
    indexed.sort(key=lambda x: x[0], reverse=True)
    reranked = [d for _, d in indexed[:RERANK_TOP_K]]
    return {"reranked_docs": reranked}

# -----------------------------
# 4. Doc Filtering (relevance threshold)
# -----------------------------
DOC_SCORE_THRESHOLD = 0.3


def doc_filtering_node(state: State) -> State:
    q = state["question"]
    docs = state["reranked_docs"]
    if not docs:
        return {"filtered_docs": []}
    pairs = [(q, d.page_content) for d in docs]
    scores = cross_encoder.predict(pairs)
    filtered = [d for d, s in zip(docs, scores) if s > DOC_SCORE_THRESHOLD]
    return {"filtered_docs": filtered if filtered else docs[:3]}

# -----------------------------
# Sentence-level decomposer (for context compression)
# -----------------------------
def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# -----------------------------
# 5. Context Compression (sentence-level filter)
# -----------------------------
class KeepOrDrop(BaseModel):
    keep: bool


filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Return keep=true only if the sentence directly helps answer the question. Output JSON only.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)


def _filter_invoke(question: str, sentence: str) -> KeepOrDrop:
    messages = filter_prompt.invoke({"question": question, "sentence": sentence}).to_messages()
    return qwen_invoke(messages, parse_json=True, output_schema=KeepOrDrop)


def context_compression_node(state: State) -> State:
    q = state["question"]
    docs = state["filtered_docs"]
    context = "\n\n".join(d.page_content for d in docs).strip()
    strips = decompose_to_sentences(context)
    kept = [s for s in strips if _filter_invoke(q, s).keep]
    compressed = "\n".join(kept).strip() if kept else context[:3000]
    return {"compressed_context": compressed}

# -----------------------------
# 6. LLM Generation (draft answer)
# -----------------------------
answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
            "If the context is empty or insufficient, say: 'I don't know.'",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)


def llm_generation_node(state: State) -> State:
    messages = answer_prompt.invoke({
        "question": state["question"],
        "context": state["compressed_context"],
    }).to_messages()
    draft = qwen_invoke(messages)
    return {"draft_answer": draft}

# -----------------------------
# 7. Hallucination Check
# -----------------------------
class HallucinationVerdict(BaseModel):
    grounded: bool
    reason: str


hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the question, the context used, and the model answer, decide if the answer is fully grounded in the context.\n"
            "Set grounded=true only if every factual claim in the answer is supported by the context.\n"
            "Output JSON only: {\"grounded\": true/false, \"reason\": \"short reason\"}",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"),
    ]
)


def hallucination_check_node(state: State) -> State:
    messages = hallucination_prompt.invoke({
        "question": state["question"],
        "context": state["compressed_context"],
        "answer": state["draft_answer"],
    }).to_messages()
    out = qwen_invoke(messages, parse_json=True, output_schema=HallucinationVerdict)
    return {"hallucination_ok": out.grounded}

# -----------------------------
# 8. Final Answer
# -----------------------------
def final_answer_node(state: State) -> State:
    if state.get("hallucination_ok"):
        final = state["draft_answer"]
    else:
        final = "I don't have enough supported information to answer that confidently."
    return {"final_answer": final}

# -----------------------------
# Build graph: User Query → Query Expansion → Hybrid Retrieval → Cross-Encoder Rerank
#              → Doc Filtering → Context Compression → LLM Generation → Hallucination Check → Final Answer
# -----------------------------
g = StateGraph(State)

g.add_node("query_expansion", query_expansion_node)
g.add_node("hybrid_retrieval", hybrid_retrieval_node)
g.add_node("cross_encoder_rerank", cross_encoder_rerank_node)
g.add_node("doc_filtering", doc_filtering_node)
g.add_node("context_compression", context_compression_node)
g.add_node("llm_generation", llm_generation_node)
g.add_node("hallucination_check", hallucination_check_node)
g.add_node("final_answer", final_answer_node)

g.add_edge(START, "query_expansion")
g.add_edge("query_expansion", "hybrid_retrieval")
g.add_edge("hybrid_retrieval", "cross_encoder_rerank")
g.add_edge("cross_encoder_rerank", "doc_filtering")
g.add_edge("doc_filtering", "context_compression")
g.add_edge("context_compression", "llm_generation")
g.add_edge("llm_generation", "hallucination_check")
g.add_edge("hallucination_check", "final_answer")
g.add_edge("final_answer", END)

app = g.compile()

# -----------------------------
# Run example
# -----------------------------
INITIAL_STATE = {
    "question": "Batch normalization vs layer normalization",
    "expanded_queries": [],
    "docs": [],
    "reranked_docs": [],
    "filtered_docs": [],
    "compressed_context": "",
    "draft_answer": "",
    "hallucination_ok": False,
    "final_answer": "",
}

res = app.invoke(INITIAL_STATE)

print("Expanded queries:", res.get("expanded_queries"))
print("Hallucination OK:", res.get("hallucination_ok"))
print("\nFINAL ANSWER:\n", res["final_answer"])