import os
import re  # Used to clean PDF extraction artifacts
import glob # Used for file searching
import logging # Used to suppress warnings from pypdf
import warnings
warnings.filterwarnings("ignore") # Suppress warnings for a cleaner output
logging.getLogger("pypdf").setLevel(logging.ERROR) # Suppress pypdf warnings about malformed PDFs

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import Tool

# CONFIGURATION CONSTANTS #
DOCUMENTS_PATH    = "data/documents"
VECTOR_STORE_PATH = "data/vector_store"
COLLECTION_NAME   = "income_domain_kb"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

# EMBEDDING INITIALIZATION #
def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# DOCUMENT LOADING AND SPLITTING #
def load_and_split_documents():
    documents = []

    pdf_files = glob.glob(os.path.join(DOCUMENTS_PATH, "**", "*.pdf"), recursive=True)
    for path in pdf_files:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
        print(f"  [PDF] {os.path.basename(path)}")

    txt_files = glob.glob(os.path.join(DOCUMENTS_PATH, "**", "*.txt"), recursive=True)
    for path in txt_files:
        loader = TextLoader(path, encoding="utf-8")
        documents.extend(loader.load())
        print(f"  [TXT] {os.path.basename(path)}")

    print(f">> Loaded {len(documents)} document page(s) total")

    # Remove the leader-dash artifacts left behind by the PDF extractor:
    for d in documents:
        d.page_content = re.sub(r"(?:leaderdash/?)+", " ", d.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Ordered from the biggest natural break to the smallest
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f">> Created {len(chunks)} chunks")
    return chunks

# CHROMA_DB VECTOR STORE CREATION AND LOADING #
def create_vector_store():
    global _vectorstore  # so the module-level handle points at the fresh store

    chunks = load_and_split_documents()
    embeddings = _get_embeddings()

    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    # Wipe the old collection so reruns don't pile up duplicates:
    Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_PATH,
    ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_PATH,
    )
    _vectorstore = vectorstore
    print(f">> Vector store created at: {VECTOR_STORE_PATH}")
    return vectorstore

def load_vector_store():
    embeddings = _get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_PATH,
    )
    print(f">> Vector store loaded from: {VECTOR_STORE_PATH}")
    return vectorstore

# RETRIEVAL TOOL FOR THE K-MOST RELEVANT CHUNKS #
try:
    _vectorstore = load_vector_store()
except Exception:
    _vectorstore = None

def retrieve_context(query: str, k: int = 5):
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vector_store()

    retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 15, "lambda_mult": 0.4},
    )

    docs = retriever.invoke(query)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return context, docs

def _retrieve_for_tool(query: str) -> str:
    context, _ = retrieve_context(query)
    return context

retrieval_tool = Tool(
    name="KnowledgeBaseRetrieval",
    func=_retrieve_for_tool,
    description=(
        "Queries the local knowledge base to answer conceptual or factual domain questions. "
        "Trigger this for inquiries regarding: UCI Adult Census features, socio-demographic trends. "
        "Stage 1 models (Logistic Regression, Random Forest, MLP), preprocessing, and PCA insights. "
        "Input: Natural language question. "
        "Output: Relevant grounding context from project documentation."
    ),
)

# INTERACTIVE TESTING #

if __name__ == "__main__":
    create_vector_store()

    queries = [
        "Which are the main factors that make the income less than 50k dollars?",
        "Does gender play a role in income?",
        "Why was Logistic Regression chosen as the best model?",
    ]

    for q in queries:
        context, docs = retrieve_context(q)
        print(f"\nUser query: {q}")
        print(f"Retrieved chunks: {docs[0].page_content[:300]}")
        print(f"Source: {os.path.basename(docs[0].metadata.get('source', '?'))}")
        print("\n")
