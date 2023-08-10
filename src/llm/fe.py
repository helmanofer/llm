import os

# import openai
import streamlit as st

if st.session_state.get("openai_api_key", None) is None:
    st.session_state["openai_api_key"] = ""


st.text_input(
    "OpenAI API Key",
    type="password",
    key="openai_api_key",
    value=st.session_state.get("openai_api_key"),
)

if st.session_state.get("openai_api_key"):
    key = st.session_state.get("openai_api_key", "")
    st.text(f"Open AI key was provided {key[0:5]}...")
    # openai.api_key = key
    os.environ["OPENAI_API_KEY"] = st.session_state.get("openai_api_key", "")
