# Module 3 — Support Assistant

## Run locally
```bash
pip install -r requirements.txt
# MOCK_LLM defaults to "1" (offline, deterministic, graded baseline) — leave it unset.
uvicorn main:app --reload
```

Then, in another terminal:
```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"query": "Do you offer free delivery?"}'

curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
```

## Run with Docker (required, graded baseline — no push needed)
```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
# then POST to http://127.0.0.1:7860/ask as above
```

## Example call transcripts

**Call 1 — triggers retrieval** (query: "Do you offer free delivery?"):
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation...","sources":["doc_01","doc_05","doc_02"],"confidence":1.0}

**Call 2 — does NOT trigger retrieval** (query: "What is the capital of France?"):
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}

## Architecture (ingestion -> embedding -> retrieval -> generation)
1. **Ingestion**: `docs/doc_01.txt` … `doc_08.txt` — Zepto's 8 policy
   documents, one per file, loaded by `build_collection()` in `main.py`.
   Each file is treated as one chunk (simple per-document chunking, fine
   given their short length).
2. **Embedding**: `embed_texts()` calls `sentence-transformers`'
   `all-MiniLM-L6-v2` model locally (no API key, no cost) to embed each
   chunk. `build_collection()` stores the 8 (id, text, embedding) triples in
   a ChromaDB in-memory collection named `zepto_policies`.
3. **Retrieval**: the LangGraph node `retrieve_and_answer` embeds the
   incoming query with the same model and calls `collection.query(...)` to
   fetch the top-3 most similar chunks via cosine similarity. This step
   always runs for real, in both MOCK_LLM states — it needs no API key.
4. **Generation**: two LangGraph nodes produce the final answer —
   `retrieve_and_answer` (for `policy_question` queries, grounded in the
   retrieved chunks) and `direct_answer` (for `general_question` queries, no
   retrieval). A third node, `classify_intent`, routes between them via a
   conditional edge (`route_from_intent`). The result is validated against
   the `AskResponse` Pydantic model (`answer`, `sources`, `confidence`)
   before FastAPI's `POST /ask` returns it.

**Where `MOCK_LLM` branches**: only inside the *generation* step of each of
the 3 nodes (`classify_intent`, `retrieve_and_answer`, `direct_answer`) —
never in ingestion, embedding, or retrieval, which always run for real.
- **Default (`MOCK_LLM` unset or `1`, graded baseline)**: `classify_intent`
  uses a keyword heuristic (see `POLICY_KEYWORDS` in `main.py`);
  `retrieve_and_answer` returns a canned `"Based on the retrieved context:
  {snippet}"` string built from the top retrieved chunk;
  `direct_answer` returns a fixed canned string. No network call to any LLM
  provider is made anywhere in this state.
- **Optional `MOCK_LLM=0` extension**: each node instead calls a real LLM
  (`_call_real_llm`, wired for Groq's free tier — set `GROQ_API_KEY`) using
  the structured prompt template in `PROMPT_TEMPLATE` (role-context-task-
  format-length skeleton, with a negative constraint and a few-shot
  example). If the LLM's output fails Pydantic validation,
  `run_graph_validated()` retries up to 2 additional times with a
  corrective instruction before returning a clearly marked error response.

## Files
- `docs/doc_01.txt` … `doc_08.txt` — the policy corpus (verbatim as specified).
- `main.py` — embedding + ChromaDB ingestion, the LangGraph graph (3 nodes +
  conditional routing), the Pydantic schema, and the FastAPI app.
- `Dockerfile` — builds and runs the FastAPI app locally on port 7860.
- `requirements.txt` — Python dependencies.
