# test_create_vector_store.py
from knowledge_base import create_vector_store

try:
    vector_store = create_vector_store("data/documents")
    print("Vector store created successfully!")
except Exception as e:
    print(f"Error: {str(e)}")