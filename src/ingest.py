import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base")
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector_store")


def build_vector_store():
    print("Loading documents...")
    loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("Building vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        persist_directory=VECTOR_DB_DIR,
    )
    print("Done! Vector store is ready.")
    return vector_store


if __name__ == "__main__":
    build_vector_store()