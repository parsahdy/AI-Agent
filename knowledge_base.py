from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from file_processor import load_documents

def create_vector_store(documents_path):
    documents = load_documents(documents_path)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("data/vector_store")
    return vector_store

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("data/vector_store", embeddings, allow_dangerous_deserialization=True)