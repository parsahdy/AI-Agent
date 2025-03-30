# test_load_documents.py
from file_processor import load_documents


directory_path = "data/documents"
try:
    documents = load_documents(directory_path)
    print(f"Loaded {len(documents)} document chunks:")
    for i, doc in enumerate(documents):
        print(f"Document {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content (first 200 chars): {doc.page_content[:200]}")
        print("-" * 50)
except Exception as e:
    print(f"Error: {str(e)}")