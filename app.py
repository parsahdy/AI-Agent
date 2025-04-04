import streamlit as st
import os
from langdetect import detect
from llm_connector import get_llm
from knowledge_base import load_vector_store, create_vector_store
from planner import create_weekly_plan
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=Warning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

st.set_page_config(page_title="Smart Academic Advisor", layout="wide")
st.title("Smart Academic Advisor")

st.markdown("""
    <style>
    .rtl {
        direction: rtl;
        text-align: right;
        font-family: 'Tahoma', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

if not os.path.exists("data/documents/persian_wikipedia_qa.txt"):
    if not os.path.exists("data/documents"):
        os.makedirs("data/documents")

    ds = load_dataset("fibonacciai/Persian-Wikipedia-QA")
    with open("data/documents/persian_wikipedia_qa.txt", "w", encoding="utf-8") as f:
        for item in ds["train"]:
            f.write(f"سوال: {item['question']}\nجواب: {item['answer']}\n\n")

@st.cache_resource
def initialize_resources():
    try:
        llm = get_llm("nous-hermes")

        tokenizer = None

        folder_path = "data/vector_store"
        index_file = os.path.join(folder_path, "index.faiss")
        if not os.path.exists(folder_path) or not os.path.exists(index_file):
            st.write("فایل ذخیره‌سازی بردار یافت نشد یا ناقص است، در حال ساخت یک فایل جدید...")
            vector_store = create_vector_store("data/documents")
        else:
            vector_store = load_vector_store()

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )

        return llm, qa_chain, tokenizer

    except Exception as e:
        st.error(f"Error loading: {str(e)}")
        import traceback
        st.error(f"Error details: {traceback.format_exc()}")
        return None, None, None

with st.spinner("Loading models... This may take a while.") :
    llm, qa_chain, tokenizer = initialize_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "weekly_plan" not in st.session_state:
    st.session_state.weekly_plan = {}

st.write("I can give you advice or create a weekly plan for you! For advice, ask your question (e.g., 'How can I focus better when studying?'). For a weekly plan, say 'Create a weekly plan' or 'I want a weekly plan for [topic]'.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="rtl">{message["content"]}</div>', unsafe_allow_html=True)

user_input = st.chat_input("Ask your question...")

weekly_plan_phrases = [
    "یک برنامه هفتگی بساز",
    "یک برنامه هفتگی می‌خواهم",
    "برنامه هفتگی برای",
    "برنامه هفتگی بساز",
    "برنامه هفتگی ایجاد کن"
]

if user_input and llm and qa_chain:
    truncated_input = user_input[:400]

    st.session_state.messages.append({"role": "user", "content": truncated_input})
    with st.chat_message("user"):
        st.markdown(f'<div class="rtl">{truncated_input}</div>', unsafe_allow_html=True)

    try:
        user_input_lower = truncated_input.lower()
        is_weekly_plan_request = any(phrase in user_input_lower for phrase in weekly_plan_phrases)

        if is_weekly_plan_request:
            with st.spinner("Creating a weekly schedule..."):
                response, week_num = create_weekly_plan(truncated_input, llm)
                st.session_state.weekly_plan[week_num] = response
        else:
            with st.spinner("Generating response..."):
                try:
                    is_persian = detect(truncated_input) == 'fa'
                except:
                    is_persian = False

                result = qa_chain.invoke({"question": truncated_input})
                if "answer" in result:
                    response = result["answer"]
                else:
                    response = "I couldn't find a suitable answer. Please clarify your question."

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(f'<div class="rtl">{response}</div>', unsafe_allow_html=True)

        st.rerun()

    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        import traceback
        st.error(f"Full error details: {traceback.format_exc()}")

tab1, tab2 = st.tabs(["weekly_plan", "conversation"])

with tab1:
    st.header("Request a weekly schedule")
    plan_input = st.text_input("Enter your request for a weekly schedule (for example: 'I want a weekly schedule for studying math.')", key="weekly_plan_input")
    if plan_input:
        with st.spinner("Creating a weekly schedule..."):
            response, week_num = create_weekly_plan(plan_input, llm)
            st.session_state.weekly_plan[week_num] = response
            st.markdown(f'<div class="rtl">{response}</div>', unsafe_allow_html=True)

with tab2:
    st.header("Conversation history")
    with st.container(height=300):
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(f'<div class="rtl">{message["content"]}</div>', unsafe_allow_html=True)
            if st.button("Delete conversation history"):
                st.session_state.messages = []
                try:
                    qa_chain.memory_clear()
                except:
                    pass
                st.rerun()
        else:
            st.info("No conversation has taken place yet. Ask your question in the chat section!")

with tab1:
    st.header("My weekly schedule")
    weeks = list(st.session_state.weekly_plan.keys())
    if weeks:
        selected_week = st.selectbox("Select a week:", weeks)
        st.markdown(f'<div class="rtl">{st.session_state.weekly_plan[selected_week]}</div>', unsafe_allow_html=True)
    else:
        st.info("The weekly schedule has not been created yet. Please request a weekly schedule from the chat section or the form above.")
