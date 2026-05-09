from fastapi import FastAPI
from pydantic import BaseModel, Field
from RAG_final import ask_rag

app = FastAPI(
    title="RAG Assistant API",
    description="FastAPI wrapper for LangGraph + FAISS + Guardrails RAG workflow",
    version="1.0.0"
)


class AskRequest(BaseModel):
    question: str = Field(
        description="User question to ask the RAG assistant",
        example="What is an AI agent?"
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    validation: str
    retries: int


@app.get("/")
def home():
    return {
        "message": "RAG Assistant API is running",
        "endpoint": "/ask",
        "method": "POST"
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    result = ask_rag(request.question)
    return result