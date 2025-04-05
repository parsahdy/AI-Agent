from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vector_store():
    try:
        # استفاده از یک مدل embedding ساده
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # ایجاد یک پایگاه دانش ساده (برای تست)
        # در عمل، باید داده‌های واقعی را اینجا بارگذاری کنید
        texts = ["این یک متن نمونه برای تست است.", "پایگاه دانش برای پاسخ به سؤالات."]
        vector_store = FAISS.from_texts(texts, embeddings)
        
        return vector_store
    except Exception as e:
        print(f"خطا در بارگذاری پایگاه دانش: {str(e)}")
        return None