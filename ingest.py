from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


from config import EMBEDDING_MODEL, CHROMA_DIR, PDF_PATH

print("Loading PDF...")
loader = PyMuPDFLoader(PDF_PATH)
docs = loader.load()

print("Chunking text...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)

print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("Storing in ChromaDB...")
db = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)
db.persist()

print("Ingestion complete!")




