# RAG Assistant (LangGraph + FAISS + Guardrails)

A production-style RAG pipeline that retrieves relevant chunks using **FAISS**, generates answers grounded in retrieved context, validates outputs, and retries safely when validation fails.

## Architecture
START → Retriever(FAISS) → Responder(LLM) → Validator → [VALID → END | INVALID → Retriever]

## Features
- ✅ Real vector search with FAISS
- ✅ Chunking + overlap for better retrieval
- ✅ Answer grounding using retrieved context
- ✅ Validation guardrail (VALID / INVALID)
- ✅ Bounded retry logic
- ✅ Works with Groq LLM

## Setup
1) Create & activate venv:
```bash
python -m venv rag_venv
rag_venv\Scripts\activate
