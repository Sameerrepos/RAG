FAISS retrieval (Top‑K chunks)
Grounded generation + validation + bounded retry

## limitations:-
- Retrieval quality depends on document quality, chunking strategy, and embedding model quality.
- Wrong or irrelevant chunks can lead to weak or incorrect answers.
- Current implementation uses a small local knowledge base.
- Local HuggingFace embedding models can be memory-heavy for free-tier cloud platforms.
- Render Free Tier deployment may fail due to memory limits when loading local embeddings.
- Next improvements can include reranking, lightweight/external embeddings, managed vector search, logging, observability, and cloud deployment
# RAG Assistant (LangGraph + FAISS + Guardrails)

A production-style RAG pipeline that retrieves relevant chunks using **FAISS**, generates answers grounded in retrieved context, validates outputs, and retries safely when validation fails.

---

## Architecture

```text
START → Retriever(FAISS) → Responder(LLM) → Validator → [VALID → END | INVALID → Retriever]
```

---

## Features

- ✅ Real vector search with FAISS
- ✅ Chunking + overlap for better retrieval
- ✅ FAISS retrieval with Top-K chunks
- ✅ Grounded generation using retrieved context
- ✅ Citations / sources for traceability
- ✅ Validation guardrail (VALID / INVALID)
- ✅ Bounded retry logic
- ✅ FastAPI backend API
- ✅ Dockerized for reproducible local deployment
- ✅ Works with Groq LLM

---

## Setup

### 1. Create and activate virtual environment

```bash
python -m venv rag_venv
rag_venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env` file

Create a `.env` file using `.env.example`:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=optional_huggingface_token_here
```

> Do not commit `.env` to GitHub.

---

## Run Locally

Run the FastAPI server:

```bash
uvicorn api:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## API Usage

### Endpoint

```http
POST http://127.0.0.1:8000/ask
```

### Request Body

```json
{
  "question": "What is an AI agent?"
}
```

### Example Response

```json
{
  "question": "What is an AI agent?",
  "answer": "An AI agent is a system that can perceive its environment, reason about actions, and act to achieve specific goals [S1].",
  "sources": [
    "[S1] score=0.3368 :: An AI agent is a system that can perceive its environment..."
  ],
  "validation": "VALID",
  "retries": 0
}
```

---

## Docker Usage

### Build the Docker image

```bash
docker build -t rag-fastapi .
```

### Run the Docker container

```bash
docker run --env-file .env -p 8000:8000 rag-fastapi
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Test API endpoint using Postman:

```http
POST http://127.0.0.1:8000/ask
```

Request body:

```json
{
  "question": "What is an AI agent?"
}
```

## Logging / Observability

The API logs request-level metadata for debugging and monitoring:

- User question
- LLM API calls
- Validation result
- Retry count
- Retrieved source count
- Response time
- HTTP status

Example log:

```text
Request completed | validation=VALID | retries=0 | sources_count=1 | response_time=0.513s
---

## 4. Commit latest logging changes

Run:

```powershell
git status
git add api.py README.md
git commit -m "Add request logging for RAG API"
git push
---
## Live Deployment

Swagger UI:
POST https://your-render-url.onrender.com/ask


## Project Highlights

- FAISS retrieval using Top-K document chunks
- Grounded generation using retrieved context
- Validation guardrails for relevance and grounding
- Bounded retry logic when validation fails
- FastAPI API layer for backend integration
- Docker containerization for reproducible deployment

---

## Limitations

- Retrieval quality depends on document quality and chunking strategy.
- Wrong or irrelevant chunks can lead to weak answers.
- Current implementation uses a small local knowledge base.
- Next improvements can include reranking, better chunking strategies, logging, and deployment to cloud.

---

## Tech Stack

- Python
- LangGraph
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLM
- FastAPI
- Uvicorn
- Docker

---

## Repository Structure

```text
RAG/
│
├── api.py
├── RAG_final.py
├── documents.txt
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env.example
├── README.md
└── archive/
```
## Deployment Note

This project runs successfully locally using Docker.

During cloud deployment testing on Render Free Tier, the service hit memory limits while loading local HuggingFace embedding models. This is expected because local embedding models and dependencies such as sentence-transformers can require more memory than lightweight free-tier instances provide.

For production deployment, recommended options include:
- Use a memory-optimized cloud instance
- Replace local embeddings with a lightweight embedding provider
- Use an external embedding API
- Move vector search to a managed service such as Azure AI Search, Pinecone, Qdrant, or similar
- Add logging, monitoring, and request-level observability before production use

---

## Author

**Sameer Shaik**  
GitHub: https://github.com/Sameerrepos
