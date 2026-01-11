import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def ingest_documents():
    data_dir = os.path.join(os.path.dirname(__file__), "../data_docs")
    persist_dir = os.path.join(os.path.dirname(__file__), "./vectorstore")

    os.makedirs(persist_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print("❌ No PDF files found in 'data/reports'.")
        return

    all_docs = []
    for pdf in pdf_files:
        loader = PyMuPDFLoader(os.path.join(data_dir, pdf))
        documents = loader.load()
        all_docs.extend(documents)
        print(f"📘 Loaded {pdf}")

    # Split long texts into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(all_docs)
    print(f"✅ Total {len(texts)} text chunks created.")

    # Use sentence-transformers model for embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    # Create or update Chroma vector store
    db = Chroma.from_documents(
        texts, embeddings, persist_directory=persist_dir)
    db.persist()
    print(f"✅ Vector store saved to {persist_dir}")


if __name__ == "__main__":
    ingest_documents()
