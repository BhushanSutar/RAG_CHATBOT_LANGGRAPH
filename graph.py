from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
from config import EMBEDDING_MODEL, CHROMA_DIR, GROQ_MODEL
from dotenv import load_dotenv
load_dotenv()

# ---------- Load Vector DB ----------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)
retriever = db.as_retriever(search_kwargs={"k": 4})

# ---------- Groq Client ----------
client = Groq()

# ---------- State ----------
class RAGState(TypedDict):
    question: str
    docs: List
    answer: str
    confidence: float

# ---------- Nodes ----------
def retrieve_node(state):
    # Changed from get_relevant_documents() to invoke()
    docs = retriever.invoke(state["question"])
    return {"docs": docs}

def relevance_check_node(state):
    if not state["docs"]:
        return {"answer": "I could not find this information in the provided document."}
    return state

def answer_node(state):
    context = "\n\n".join([d.page_content for d in state["docs"]])

    prompt = f"""
You are a helpful AI assistant.
Answer ONLY using the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{state['question']}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return {"answer": response.choices[0].message.content}

def confidence_node(state):
    confidence = round(min(0.6 + 0.1 * len(state["docs"]), 0.95), 2)
    return {"confidence": confidence}

# ---------- Build Graph ----------
graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("check", relevance_check_node)
graph.add_node("answer", answer_node)
graph.add_node("confidence", confidence_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "check")
graph.add_edge("check", "answer")
graph.add_edge("answer", "confidence")
graph.set_finish_point("confidence")

rag_app = graph.compile()