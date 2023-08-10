import os
import tempfile
from typing import List

# import openai
import streamlit as st

from common.models import QuestionAnswer
from services.search_store_service import ChromaVectorStore


# if os.environ.get("OPENAI_API_KEY", ""):
#     openai.openai_api_key = os.environ.get("OPENAI_API_KEY", "")


@st.cache_resource
def get_cached_search_service(name_=None):
    if not name_:
        name_ = st.session_state.index_name
    index_ = ChromaVectorStore(name_)
    return index_


tab2, tab1 = st.tabs(["Use existing data", "Load new data"])
with tab1:
    name = "_".join(st.text_input("Name this index").lower().split())
    uploaded_file = st.file_uploader("Choose a file")
    sp = st.text_input(
        "Split text to sentences by token or split by chunk of chars", value=1000
    )
    ix = st.button("Index")
    txt = []

    if uploaded_file and sp and name and ix:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file_name = os.path.join(tmpdir, uploaded_file.name)
            with open(tmp_file_name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            search_service = get_cached_search_service(name)
            txt: List[str] = search_service.split_text(tmpdir, sp)
            st.table(txt[0:10])

    if txt:
        with st.spinner("Wait for indexing"):
            search_service = get_cached_search_service(name)

            search_service.store_documents(txt)
        st.success("Done!")


def clear():
    st.session_state.q = ""


with tab2:
    indexes = ChromaVectorStore.list_indexes()
    index_name = st.selectbox(
        "Choose an index to query", indexes, key="index_name", on_change=clear
    )

    st.text_input("Ask a question", key="q")

    if st.session_state.q and st.session_state.index_name:
        search_service = get_cached_search_service()
        with st.spinner("Wait for query result"):
            a: QuestionAnswer = search_service.query(st.session_state.q)
        st.success("Done!")
        st.markdown(a.answer)
        data = []
        for node in a.documents:
            nn = {"text": node.text, "score": node.score}
            data.append(nn)
        st.table(data)
