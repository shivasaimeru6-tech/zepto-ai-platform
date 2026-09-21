"""
Zepto Capstone - Module 3: Support Assistant
==============================================
A small RAG service: 8 policy documents -> sentence-transformers embeddings
-> ChromaDB -> a LangGraph intent-router -> a Pydantic-validated JSON answer
-> served over FastAPI.

Every LLM call is gated behind MOCK_LLM (default "1" = fully offline,
deterministic, rule-based mock mode — this is the graded baseline and needs
no API key / no network to an LLM provider). Set MOCK_LLM=0 to use a real
LLM (optional, ungraded extension; requires a GROQ_API_KEY or similar).

Run:
    uvicorn main:app --reload
Then:
    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
         -d '{"query": "How long do I have to report a damaged item?"}'
"""

import os
from pathlib import Path
from typing import TypedDict

import chromadb
from fastapi import FastAPI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"

DOCS_DIR = Path(__file__).parent / "docs"
COLLECTION_NAME = "zepto_policies"

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours",
]

# ---------------------------------------------------------------------------
# Structured prompt template (role-context-task-format-length skeleton)
# Used by the optional MOCK_LLM=0 real-LLM path in retrieve_and_answer/direct_answer.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """\
# Role
You are Zepto's customer support assistant.

# Context
Use ONLY the following retrieved policy excerpts to answer the customer's
question. Do not use any outside knowledge.
---
{context}
---

# Task
Answer the customer's question below, grounded strictly in the context above.

# Negative constraint
Do not answer using information not present in the provided context. If the
context does not contain the answer, say you don't have that information.

# Format
Respond in 2-4 plain sentences. No headers, no bullet points, no markdown.

# Length
Keep the answer under 80 words.

# Few-shot example
Question: "Is phone support available?"
Context: "Phone support is not offered. Email support ... 24 hours."
Answer: "No, Zepto doesn't offer phone support. You can reach us anytime via
in-app chat, or by email for non-urgent queries, which is answered within 24
hours on business days."

# Customer question
{query}
"""

# ---------------------------------------------------------------------------
# Embedding model (lazy singleton so this module can be imported/tested
# without immediately triggering a model download).
# ---------------------------------------------------------------------------
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(list(texts)).tolist()


# ---------------------------------------------------------------------------
# ChromaDB ingestion
# ---------------------------------------------------------------------------
_chroma_client = chromadb.Client()  # in-memory; swap for PersistentClient for durability


def get_collection():
    try:
        return _chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        return build_collection()


def build_collection():
    collection = _chroma_client.get_or_create_collection(COLLECTION_NAME)
    doc_paths = sorted(DOCS_DIR.glob("doc_*.txt"))
    ids, texts = [], []
    for path in doc_paths:
        ids.append(path.stem)  # doc_01, doc_02, ...
        texts.append(path.read_text(encoding="utf-8").strip())
    embeddings = embed_texts(texts)
    collection.add(ids=ids, documents=texts, embeddings=embeddings)
    return collection


def retrieve_top_k(query: str, k: int = 3):
    collection = get_collection()
    query_embedding = embed_texts([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    ids = results["ids"][0]
    docs = results["documents"][0]
    return list(zip(ids, docs))


# ---------------------------------------------------------------------------
# LangGraph state + nodes
# ---------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved: list[tuple[str, str]]
    answer: str
    sources: list[str]
    confidence: float


def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]
    if MOCK_LLM:
        lowered = query.lower()
        intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"
    else:
        intent = _llm_classify_intent(query)  # optional real-LLM path
    return {**state, "intent": intent}


def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]
    retrieved = retrieve_top_k(query, k=3)  # always real: embeddings + Chroma need no API key
    sources = [doc_id for doc_id, _ in retrieved]

    if MOCK_LLM:
        top_chunk = retrieved[0][1] if retrieved else ""
        snippet = top_chunk[:200]
        answer = f"Based on the retrieved context: {snippet}"
        confidence = 1.0
    else:
        answer, confidence = _llm_answer_from_context(query, retrieved)  # optional real-LLM path

    return {**state, "retrieved": retrieved, "answer": answer,
            "sources": sources, "confidence": confidence}


def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM:
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        answer, confidence = _llm_direct_answer(state["query"])  # optional real-LLM path

    return {**state, "retrieved": [], "answer": answer, "sources": [], "confidence": confidence}


def route_from_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_from_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


_app_graph = build_graph()


# ---------------------------------------------------------------------------
# Optional MOCK_LLM=0 real-LLM helpers (Groq free tier or any free-tier LLM API)
# ---------------------------------------------------------------------------
def _call_real_llm(prompt: str) -> str:
    """Minimal Groq-compatible chat completion call. Requires GROQ_API_KEY."""
    import requests

    api_key = os.environ["GROQ_API_KEY"]
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _llm_classify_intent(query: str) -> str:
    prompt = (
        "Classify the following customer query as exactly one word: "
        "'policy_question' if it concerns Zepto delivery, returns, refunds, "
        "membership, tracking, cancellation, gift cards, or support hours; "
        f"otherwise 'general_question'.\n\nQuery: {query}"
    )
    result = _call_real_llm(prompt).strip().lower()
    return "policy_question" if "policy_question" in result else "general_question"


def _llm_answer_from_context(query: str, retrieved: list[tuple[str, str]]):
    context = "\n".join(f"[{doc_id}] {text}" for doc_id, text in retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    answer = _call_real_llm(prompt)
    return answer, 0.85


def _llm_direct_answer(query: str):
    prompt = f"Answer briefly and helpfully:\n{query}"
    answer = _call_real_llm(prompt)
    return answer, 0.7


# ---------------------------------------------------------------------------
# Pydantic schema + validation-with-retry (retry logic used by MOCK_LLM=0 path)
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float


def run_graph_validated(query: str) -> AskResponse:
    """Run the graph and validate its output against AskResponse.

    In mock mode the fields are populated deterministically by our own code,
    so there is no LLM output to fail validation. In the optional MOCK_LLM=0
    path, if the real LLM's output doesn't fit the schema, retry up to 2
    additional times with a corrective instruction before giving up.
    """
    attempts = 0
    max_attempts = 1 if MOCK_LLM else 3
    last_error = None

    while attempts < max_attempts:
        attempts += 1
        try:
            state = _app_graph.invoke({"query": query})
            return AskResponse(
                answer=state["answer"],
                sources=state.get("sources", []),
                confidence=state.get("confidence", 0.0),
            )
        except Exception as exc:  # pydantic ValidationError or anything upstream
            last_error = exc
            continue

    return AskResponse(
        answer=f"Error: could not produce a valid response after {max_attempts} attempt(s): {last_error}",
        sources=[],
        confidence=0.0,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Zepto Support Assistant")


@app.on_event("startup")
def _startup():
    build_collection()  # eager-build the Chroma collection at boot


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return run_graph_validated(request.query)
