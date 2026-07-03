import os
import sys
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
from rag_chain import answer_query
from ingest import build_vector_store, VECTOR_DB_DIR

load_dotenv()

# Auto-build vector store if not yet built
if not os.path.exists(VECTOR_DB_DIR) or not os.listdir(VECTOR_DB_DIR):
    with st.spinner("Setting up knowledge base for the first time..."):
        build_vector_store()

st.set_page_config(
    page_title="AZON Parent Support",
    page_icon="🌿",
    layout="centered",
)

# Sidebar
with st.sidebar:
    st.image("https://azon.my/templates/rt_horizon/custom/images/logo.png", width=150)
    st.markdown("### AZON Parent Support")
    st.write(
        "This is a safe space for parents to share "
        "their feelings and concerns about their child's "
        "wellbeing. You are not alone."
    )
    st.markdown("---")
    st.markdown("**Need immediate help?**")
    st.markdown("📞 Helpline HEAL: **15555**")
    st.markdown("---")
    st.markdown("**About AZON**")
    st.markdown("[Visit our website](https://azon.my)")

# API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error(
        "API key not found. Please create a .env file "
        "with your GOOGLE_API_KEY."
    )
    st.stop()

# Header
st.title("🌿 AZON Parent Support")
st.caption(
    "A safe space to share how you're feeling. "
    "We're here to listen."
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello, and welcome. 💚\n\n"
                "I'm here to listen and support you. "
                "Parenting a child who is going through difficulties "
                "can be incredibly hard, and it's okay to not be okay.\n\n"
                "How are you feeling today? Please share whatever "
                "is on your mind — there's no right or wrong thing to say here."
            ),
        }
    ]

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Share what's on your mind...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                # result = answer_query(user_input, api_key)
                # st.markdown(result["answer"])
                # st.session_state.messages.append(
                #     {"role": "assistant", "content": result["answer"]}
                # )
                
                #以下是with Sources retrieved from knowledge base框框的code，显示chatbot检索到的来源文件
                result = answer_query(user_input, api_key)
                st.markdown(result["answer"])
                if result.get("sources"):
                    with st.expander("📚 Sources retrieved from knowledge base"):
                        for s in result["sources"]:
                            st.write(f"• {s}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"]}
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")