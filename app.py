import streamlit as st
import os
from llm_connector import get_llm
from knowledge_base import load_vector_store, create_vector_store
from planner import create_weekly_plan
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Intelligent Student Counseling Agent", layout="wide")
st.title("Intelligent Student Counseling Agent")

@st.cache_resource
def initialize_resources():
    try:
        llm = get_llm("nous-hermes")  
        folder_path = "data/vector_store"
        index_file = os.path.join(folder_path, "index.faiss")
        if not os.path.exists(folder_path) or not os.path.exists(index_file):
            st.write("Vector store not found or incomplete, creating a new one...")
            vector_store = create_vector_store("data/documents")
        else:
            vector_store = load_vector_store()

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )

        return llm, qa_chain
    except Exception as e:
        st.error(f"Error in loading: {str(e)}")
        return None, None
    
llm, qa_chain = initialize_resources()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "weekly_plan" not in st.session_state:
    st.session_state.weekly_plan = {}

st.write("I can give you advice or create a weekly plan for you! For advice, ask your question (e.g. 'How can I be more focused when studying?'). For a weekly plan, say 'Create a weekly plan' or 'I want a weekly plan for [subject]'.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask your question...")

weekly_plan_phrases = [ 
    "Make a weekly schedule",
    "Make a weekly plan",
    "I want a weekly schedule",
    "I want a weekly paln",
    "Weekly schedule for",
    "Weekly plan for",
    "I want a weekly schedule",
    "I want a weekly plan",
    "weekly plan create",
    "create a weekly plan"
]

if user_input and llm and qa_chain:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        user_input_lower = user_input.lower()
        is_weekly_plan_request = any(phrase in user_input_lower for phrase in weekly_plan_phrases)

        if is_weekly_plan_request:
            with st.spinner("Creating a weekly schedule..."):
                response, week_num = create_weekly_plan(user_input, llm)
                st.session_state.weekly_plan[week_num] = response
        else:
            with st.spinner("Generating response..."):
                user_input = f"لطفاً به زبان فارسی پاسخ بده: {user_input}"
                result = qa_chain({"question": user_input})
                response = result["answer"]


        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        st.rerun()

    except Exception as e:
        st.error(f"Error processing question: {str(e)}")


tab1, tab2 = st.tabs(["weekly_plan", "conversation"])


with tab1:
    st.header("request weekly plan")
    plan_input = st.text_input("Enter your request for a weekly schedule (e.g., 'Create a weekly schedule for studying math.')", key="weekly_plan_input")
    if plan_input:
        with st.spinner("Creating a weekly schedule..."):
            response, week_num = create_weekly_plan(plan_input, llm)
            st.session_state.weekly_plan[week_num] = response
            st.markdown(response)


with tab2:
    st.header("Chat history")
    with st.container(height=300):
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            if st.button("Delete chat history"):
                st.session_state.messages = []
                qa_chain.memory.clear()
                st.rerun()
        else:
            st.info("No conversations have taken place yet. Ask your question in the chat section!")


with tab1:
    st.header("My weekly plan")
    weeks = list(st.session_state.weekly_plan.keys())
    if weeks:
        selected_week = st.selectbox("Choose a week:", weeks)
        st.markdown(st.session_state.weekly_plan[selected_week])
    else:
        st.info("The weekly schedule has not been created yet. Please request a weekly schedule in the chat section or the form above.")