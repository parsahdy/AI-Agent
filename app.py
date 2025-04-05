import streamlit as st
from transformers import pipeline
from planner import create_weekly_plan  
from knowledge_base import load_vector_store
from langdetect import detect

st.set_page_config(page_title="Smart Academic Advisor", layout="wide")
st.title("Smart Academic Advisor")

st.markdown("""
    <style>
    .rtl {
        direction: rtl;
        text-align: right;
        font-family: 'Tahoma', sans-serif;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# مقداردهی اولیه متغیرهای حالت جلسه
if "messages" not in st.session_state:
    st.session_state.messages = []

if "weekly_plan" not in st.session_state:
    st.session_state.weekly_plan = {}

# تعریف مدل با تنظیمات مناسب
try:
    llm = pipeline(
        "text-generation",
        model="HooshvareLab/gpt2-fa",
        tokenizer="HooshvareLab/gpt2-fa",
        max_length=512,
        pad_token_id=5
    )
except Exception as e:
    st.error(f"خطا در بارگذاری مدل: {str(e)}")
    llm = None

vector_store = load_vector_store()

# نمایش پیام‌های قبلی
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='rtl'>{msg['content']}</div>", unsafe_allow_html=True)

user_input = st.chat_input("سؤالت را بپرس...")

if user_input:
    with st.chat_message("user"):
        st.markdown(f"<div class='rtl'>{user_input}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    try:
        is_farsi = detect(user_input) == "fa"
    except:
        is_farsi = True
    
    with st.spinner("در حال تولید پاسخ..."):
        try:
            if llm is None:
                raise ValueError("مدل LLM بارگذاری نشده است. لطفاً بررسی کنید.")

            if "برنامه هفتگی" in user_input.lower() or "برنامه‌ریزی" in user_input.lower():
                response_text, week_num = create_weekly_plan(user_input, llm)
                st.session_state.weekly_plan[week_num] = response_text
            else:
                docs = vector_store.similarity_search(user_input, k=3)
                context = "\n".join([doc.page_content for doc in docs])
                prompt = f"سؤال: {user_input}\n\nاطلاعات مرتبط:\n{context}\n\nپاسخ مفصل:"
                
                response_full = llm(
                    prompt,
                    max_new_tokens=200,
                    do_sample=True
                )
                
                print(f"نوع response_full: {type(response_full)}")
                print(f"مقدار response_full: {response_full}")
                
                if isinstance(response_full, list) and len(response_full) > 0 and isinstance(response_full[0], dict):
                    response_full = response_full[0].get("generated_text", "")
                else:
                    response_full = str(response_full)
                
                if len(response_full) > len(prompt):
                    response_text = response_full[len(prompt):].strip()
                else:
                    response_text = response_full.strip()
                
        except Exception as e:
            st.error(f"خطا در تولید پاسخ: {str(e)}")
            response_text = "متأسفانه در تولید پاسخ مشکلی پیش آمد. لطفاً دوباره تلاش کنید."
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(f"<div class='rtl'>{response_text}</div>", unsafe_allow_html=True)
    
    st.rerun()

tab1, tab2 = st.tabs(["weekly_plan", "conversation"])

with tab1:
    st.header("درخواست برنامه هفتگی")
    plan_input = st.text_input("درخواست خود برای برنامه هفتگی را وارد کنید (برای مثال: 'یک برنامه هفتگی برای مطالعه ریاضی می‌خواهم.')", key="weekly_plan_input")
    if plan_input:
        with st.spinner("در حال ایجاد برنامه هفتگی..."):
            if llm is None:
                st.error("مدل LLM بارگذاری نشده است. لطفاً بررسی کنید.")
            else:
                response, week_num = create_weekly_plan(plan_input, llm)
                st.session_state.weekly_plan[week_num] = response
                st.markdown(f'<div class="rtl">{response}</div>', unsafe_allow_html=True)

    st.header("تاریخچه گفتگو")
    if st.session_state.weekly_plan:
        selected = st.selectbox("هفته مورد نظر را انتخاب کنید:", list(st.session_state.weekly_plan.keys()))
        st.markdown(f'<div class="rtl">{st.session_state.weekly_plan[selected]}</div>', unsafe_allow_html=True)
    else:
        st.info("هنوز برنامه‌ای تولید نشده است.")

with tab2:
    st.header("تاریخچه گفتگو")
    if st.session_state.messages:
        if st.button("پاک کردن تاریخچه"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("هنوز گفتگویی انجام نشده است.")