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

---

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

---

## Author

**Sameer Shaik**  
GitHub: https://github.com/Sameerrepos
