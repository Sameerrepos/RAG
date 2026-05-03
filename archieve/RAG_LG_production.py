from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Prefer the new package if installed; fallback otherwise
try:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# =====================================================
# 1) STATE
# =====================================================

class State(TypedDict):
    user_query: str
    retrieved_context: str
    sources: List[str]
    response: str
    validation: str
    retries: int  # counts ONLY actual failures (INVALID) that trigger re-try


# =====================================================
# 2) LLM (Groq)
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
)


# =====================================================
# 3) VECTOR STORE (FAISS)
# =====================================================

# Load docs
loader = TextLoader("documents.txt", encoding="utf-8")
raw_docs = loader.load()

# Chunk docs (important for good retrieval)
splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=60)
docs = splitter.split_documents(raw_docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS index
vector_store = FAISS.from_documents(docs, embeddings)


# =====================================================
# 4) NODES
# =====================================================

def retriever_node(state: State) -> State:
    """Retrieve top-k chunks using FAISS and build a context + sources list."""
    query = state["user_query"]

    # Retrieve with scores for transparency
    results = vector_store.similarity_search_with_score(query, k=3)

    sources: List[str] = []
    context_parts: List[str] = []

    for i, (doc, score) in enumerate(results, start=1):
        chunk = doc.page_content.strip()
        preview = (chunk[:220] + "...") if len(chunk) > 220 else chunk

        sources.append(f"[S{i}] score={score:.4f} :: {preview}")
        context_parts.append(f"[S{i}] {chunk}")

    retrieved_context = "\n\n".join(context_parts)

    return {
        **state,
        "retrieved_context": retrieved_context,
        "sources": sources,
    }


def responder_node(state: State) -> State:
    """Generate answer grounded ONLY in retrieved context."""
    prompt = f"""
You are a reliable assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say:
"I don't know based on the provided documents."

Context:
{state['retrieved_context']}

User Question:
{state['user_query']}
"""
    answer = llm.invoke(prompt).content.strip()

    return {
        **state,
        "response": answer,
    }


def validator_node(state: State) -> State:
    """
    Validate BOTH:
    1) Relevance (answers the user_query)
    2) Grounding (supported by retrieved_context; no new facts)
    """
    prompt = f"""
You are a strict validator.

Rules:
- The answer must directly address the user question.
- The answer must be supported by the context.
- The answer must NOT add facts outside the context.

User Question:
{state['user_query']}

Context:
{state['retrieved_context']}

Answer:
{state['response']}

Reply ONLY with one word: VALID or INVALID
"""
    verdict_raw = llm.invoke(prompt).content.strip().upper()

    verdict = "VALID" if "VALID" in verdict_raw else "INVALID"

    # Increment retries ONLY when INVALID (this becomes true retry count)
    retries = state["retries"] + (1 if verdict == "INVALID" else 0)

    return {
        **state,
        "validation": verdict,
        "retries": retries,
    }


# =====================================================
# 5) ROUTER (Re-retrieve on INVALID)
# =====================================================

MAX_RETRIES = 2

def router(state: State):
    if state["validation"] == "VALID":
        return END
    if state["retries"] >= MAX_RETRIES:
        return END
    # INVALID and retries left -> go back to retriever (stronger than retrying responder only)
    return "retriever"


# =====================================================
# 6) BUILD GRAPH
# =====================================================

graph = StateGraph(State)

graph.add_node("retriever", retriever_node)
graph.add_node("responder", responder_node)
graph.add_node("validator", validator_node)

graph.set_entry_point("retriever")
graph.add_edge("retriever", "responder")
graph.add_edge("responder", "validator")

graph.add_conditional_edges(
    "validator",
    router,
    {
        "retriever": "retriever",
        END: END,
    },
)

app = graph.compile()


# =====================================================
# 7) RUN
# =====================================================

if __name__ == "__main__":
    output = app.invoke({
        "user_query": "What is an AI agent?",
        "retrieved_context": "",
        "sources": [],
        "response": "",
        "validation": "",
        "retries": 0,
    })

    print("\n--- RESPONSE ---")
    print(output["response"])

    print("\n--- SOURCES (Top Retrieved Chunks) ---")
    for s in output["sources"]:
        print(s)

    print("\n--- VALIDATION ---")
    print(output["validation"])

    print("\n--- RETRIES (only counts INVALID loops) ---")
    print(output["retries"])