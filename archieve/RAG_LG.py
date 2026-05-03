from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# =====================================================
# STATE DEFINITION
# =====================================================

class State(TypedDict):
    user_query: str
    intent: str
    retrieved_docs: str
    response: str
    validation: str
    retries: int


# =====================================================
# LLM CONFIGURATION
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
)


# =====================================================
# BUILD VECTOR STORE (FAISS)
# =====================================================

loader = TextLoader("documents.txt")
documents = loader.load()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(documents, embeddings)


# =====================================================
# PLANNER NODE (INTENT)
# =====================================================

def planner_node(state: State) -> State:
    intent = llm.invoke(
        f"""
Identify the user's intent clearly.

User query:
{state['user_query']}
"""
    ).content.strip()

    return {
        "user_query": state["user_query"],
        "intent": intent,
        "retrieved_docs": "",
        "response": "",
        "validation": "",
        "retries": 0
    }


# =====================================================
# RETRIEVER NODE (REAL RAG)
# =====================================================

def retriever_node(state: State) -> State:
    query = state["user_query"]

    docs = vector_store.similarity_search(query, k=2)
    retrieved_context = "\n".join(doc.page_content for doc in docs)

    return {
        **state,
        "retrieved_docs": retrieved_context
    }


# =====================================================
# RESPONDER NODE (GROUNDED GENERATION)
# =====================================================

def responder_node(state: State) -> State:
    response = llm.invoke(
        f"""
Answer the question using ONLY the context below.

Context:
{state['retrieved_docs']}

User Question:
{state['user_query']}
"""
    ).content.strip()

    return {
        **state,
        "response": response
    }


# =====================================================
# VALIDATOR NODE (GUARDRAILS)
# =====================================================

def validator_node(state: State) -> State:
    validation = llm.invoke(
        f"""
Validate the answer based on these rules:
- Must be supported by the provided context
- Must be relevant and clear
- Must not add external facts

Answer:
{state['response']}

Context:
{state['retrieved_docs']}

Reply ONLY with VALID or INVALID.
"""
    ).content.strip()

    return {
        **state,
        "validation": validation,
        "retries": state["retries"] + 1
    }


# =====================================================
# RETRY LOGIC
# =====================================================

MAX_RETRIES = 2

def retry_router(state: State):
    if "VALID" in state["validation"].upper():
        return END
    if state["retries"] >= MAX_RETRIES:
        return END
    return "responder"


# =====================================================
# BUILD LANGGRAPH WORKFLOW
# =====================================================

graph = StateGraph(State)

graph.add_node("planner", planner_node)
graph.add_node("retriever", retriever_node)
graph.add_node("responder", responder_node)
graph.add_node("validator", validator_node)

graph.set_entry_point("planner")

graph.add_edge("planner", "retriever")
graph.add_edge("retriever", "responder")
graph.add_edge("responder", "validator")

graph.add_conditional_edges(
    "validator",
    retry_router,
    {
        "responder": "responder",
        END: END
    }
)

app = graph.compile()


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":
    output = app.invoke({
        "user_query": "What is an AI agent?"
    })

    print("\n--- RESPONSE ---")
    print(output["response"])

    print("\n--- VALIDATION ---")
    print(output["validation"])

    print("\n--- RETRIES ---")
    print(output["retries"])
