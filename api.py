from fastapi import FastAPI
from pydantic import BaseModel, Field
from RAG_final import ask_rag
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

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
    start_time = time.time()

    logger.info(f"Received question: {request.question}")

    try:
        result = ask_rag(request.question)

        response_time = round(time.time() - start_time, 3)

        logger.info(
            f"Request completed | "
            f"validation={result['validation']} | "
            f"retries={result['retries']} | "
            f"sources_count={len(result['sources'])} | "
            f"response_time={response_time}s"
        )

        return result

    except Exception as e:
        response_time = round(time.time() - start_time, 3)

        logger.error(
            f"Request failed | "
            f"question={request.question} | "
            f"error={str(e)} | "
            f"response_time={response_time}s"
        )

        raise

