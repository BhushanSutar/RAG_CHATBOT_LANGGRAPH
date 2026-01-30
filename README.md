# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangGraph that answers questions strictly from the **Agentic AI eBook**. This project demonstrates advanced RAG implementation with state management, document retrieval, and confidence scoring.

## Features

**PDF Ingestion Pipeline** - Automated document loading, chunking, and embedding generation  
**LangGraph Workflow** - State-based orchestration with multiple processing nodes  
**Vector Database** - ChromaDB for efficient semantic search  
**Grounded Answers** - Responses strictly based on the provided eBook  
**Context Retrieval** - Returns top-4 relevant document chunks  
**Confidence Scoring** - Dynamic confidence calculation based on retrieval quality  
**Interactive UI** - Streamlit-based chat interface  

---

## Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                      (Streamlit App)                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH WORKFLOW                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ RETRIEVE │───▶│  CHECK   │───▶│  ANSWER  │───▶│CONFIDENCE│ │
│  │   NODE   │    │   NODE   │    │   NODE   │    │   NODE   │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│       │               │                │                │       │
│       ▼               ▼                ▼                ▼       │
│  [Get Docs]    [Validate]      [Generate]        [Score]       │
└─────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌──────────────────────┐           ┌─────────────────────┐
│   VECTOR DATABASE    │           │    LLM (Groq)       │
│    (ChromaDB)        │           │  llama-3.3-70b      │
│                      │           └─────────────────────┘
│ ┌────────────────┐   │
│ │  Embeddings    │   │
│ │ (HuggingFace)  │   │
│ └────────────────┘   │
└──────────────────────┘
         ▲
         │
┌──────────────────────┐
│  INGESTION PIPELINE  │
│                      │
│  PDF → Chunks →      │
│  Embeddings → Store  │
└──────────────────────┘
```



## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangGraph | State-based workflow management |
| **Vector DB** | ChromaDB | Document storage & semantic search |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Text vectorization |
| **LLM** | Groq (llama-3.3-70b-versatile) | Answer generation |
| **Framework** | LangChain | Document processing & retrieval |
| **UI** | Streamlit | Interactive chat interface |
| **PDF Processing** | PyMuPDF | Document loading |

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Groq API Key 

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/agentic-ai-rag-chatbot.git
cd agentic-ai-rag-chatbot
```

### Step 2: Create Virtual Environment
```bash
python -m venv rag
source rag/bin/activate  # On Windows: rag\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Step 5: Run Ingestion Pipeline
```bash
python ingest.py
```
**Output:** Creates `chroma_db/` directory with embedded documents

### Step 6: Launch Chatbot
```bash
streamlit run app.py
```
**Access:** Open browser at `http://localhost:8501`

---

## Usage

### Running the Chatbot

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Ask questions** in the text input field

3. **View results:**
   - **Answer** - LLM-generated response
   - **Retrieved Context** - Source chunks (expandable)
   - **Confidence Score** - Retrieval quality metric

### Re-ingesting Documents

If you update the PDF or want to re-index:
```bash
python ingest.py
```

---

## 💬 Sample Queries

Here are 6 sample queries you can test:

### Query 1: **Definition Question**
```
What is agentic AI?
```
**Expected Output:** Definition and explanation from the eBook  
**Confidence:** ~0.9

---

### Query 2: **Use Case Question**
```
What are the key applications of agentic AI?
```
**Expected Output:** List of applications mentioned in the eBook  
**Confidence:** ~0.8

---

### Query 3: **Technical Question**
```
How do agents differ from traditional AI systems?
```
**Expected Output:** Comparative explanation from the document  
**Confidence:** ~0.85

---

### Query 4: **Implementation Question**
```
What are the core components of an agentic system?
```
**Expected Output:** Architectural components discussed in eBook  
**Confidence:** ~0.9

---

### Query 5: **Challenge Question**
```
What are the challenges in building agentic AI systems?
```
**Expected Output:** Challenges outlined in the document  
**Confidence:** ~0.8

---

### Query 6: **Out-of-Scope Question**
```
Who won the 2024 FIFA World Cup?
```
**Expected Output:** "I could not find this information in the provided document."  
**Confidence:** 0.6 (fallback)

---

## Project Structure

```
agentic-ai-rag-chatbot/
│
├── app.py                  # Streamlit UI application
├── graph.py                # LangGraph workflow definition
├── ingest.py               # PDF ingestion & embedding pipeline
├── config.py               # Configuration settings
├── .env                    # Environment variables (API keys)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── ARCHITECTURE.md         # Detailed architecture documentation
│
├── Ebook-Agentic-AI.pdf    # Data 
│  
│
├── chroma_db/              # Vector database (auto-generated)
│   ├── chroma.sqlite3
│   └── [embedding files]
│
└── rag/                    # Virtual environment (not in repo)
```

---

## Developer

**Bhushan Sutar**  
Email: bhushansutar1904@gmail.com  
LinkedIn: https://linkedin.com/in/bhushansutar 
GitHub: https://github.com/BhushanSutar


## License

This project is created as an assignment submission for the AI Engineer Intern role.

---





