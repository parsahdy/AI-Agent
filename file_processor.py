from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_documents(directory_path):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    documents = []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                documents.extend(docs)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                documents.extend(docs)
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            continue

    if not documents:
        raise ValueError(f"No PDF or DOCX files found in directory: {directory_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_documents(documents)