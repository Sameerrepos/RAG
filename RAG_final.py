from typing import TypedDict, List
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter


# =====================================================
# LOAD ENV
# =====================================================
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)


# =====================================================
# CONFIG
# =====================================================
INDEX_DIR = "faiss_index_fastembed"
TOP_K = 3
CHUNK_SIZE = 350
CHUNK_OVERLAP = 60
MAX_RETRIES = 2


# =====================================================
# STATE
# =====================================================
class State(TypedDict):
    user_query: str
    retrieved_context: str
    sources: List[str]
    response: str
    validation: str
    retries: int


# =====================================================
# LLM (Groq)
# =====================================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
)


# =====================================================
# VECTOR STORE (FAISS) - Build once, load later
# =====================================================
def build_or_load_vector_store() -> FAISS:
    """
    Builds or loads FAISS vector store using FastEmbed embeddings.

    Note:
    We use a new index folder: faiss_index_fastembed
    because old FAISS indexes created with HuggingFaceEmbeddings
    should not be reused with FastEmbedEmbeddings.
    """

    embeddings = FastEmbedEmbeddings()

    try:
        try:
            return FAISS.load_local(
                INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except TypeError:
            return FAISS.load_local(INDEX_DIR, embeddings)

    except Exception:
        loader = TextLoader("documents.txt", encoding="utf-8")
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        docs = splitter.split_documents(raw_docs)

        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(INDEX_DIR)

        return vector_store


vector_store = build_or_load_vector_store()


# =====================================================
# NODES
# =====================================================
def retriever_node(state: State) -> State:
    query = state["user_query"]

    results = vector_store.similarity_search_with_score(query, k=TOP_K)

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
    prompt = f"""
You are a reliable assistant.

Answer the user question using ONLY the context below.
You MUST cite sources like [S1], [S2] for each key statement.
If the answer is not in the context, reply:
"I don't know based on the provided documents."

Context:
{state['retrieved_context']}

User Question:
{state['user_query']}
"""

    answer = llm.invoke(prompt).content.strip()

    return {
        **state,
        "response": answer
    }


def validator_node(state: State) -> State:
    prompt = f"""
You are a strict validator.

Rules:
- Answer must address the user question.
- Answer must be supported by context.
- Answer must NOT introduce new facts beyond context.

User Question:
{state['user_query']}

Context:
{state['retrieved_context']}

Answer:
{state['response']}

Reply ONLY with: VALID or INVALID
"""

    raw = llm.invoke(prompt).content.strip().upper()
    verdict = "VALID" if "VALID" in raw else "INVALID"

    new_retries = state["retries"] + (1 if verdict == "INVALID" else 0)

    return {
        **state,
        "validation": verdict,
        "retries": new_retries
    }


# =====================================================
# ROUTER
# =====================================================
def router(state: State):
    if state["validation"] == "VALID":
        return END

    if state["retries"] >= MAX_RETRIES:
        return END

    return "retriever"


# =====================================================
# BUILD GRAPH
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
        END: END
    }
)

app = graph.compile()


# =====================================================
# API FUNCTION FOR FastAPI
# =====================================================
def ask_rag(user_query: str):
    output = app.invoke({
        "user_query": user_query,
        "retrieved_context": "",
        "sources": [],
        "response": "",
        "validation": "",
        "retries": 0,
    })

    return {
        "question": user_query,
        "answer": output["response"],
        "sources": output["sources"],
        "validation": output["validation"],
        "retries": output["retries"],
    }


# =====================================================
# CLI RUN
# =====================================================
if __name__ == "__main__":
    output = ask_rag("What is an AI agent?")

    print("\n--- RESPONSE (with citations) ---")
    print(output["answer"])

    print("\n--- SOURCES (retrieved) ---")
    for source in output["sources"]:
        print(source)

    print("\n--- VALIDATION ---")
    print(output["validation"])

    print("\n--- RETRIES (counts INVALID loops) ---")
    print(output["retries"])