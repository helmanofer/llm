import os
import tempfile
from typing import List
import streamlit as st

from slse import SLSE


slse = SLSE()


@st.cache_resource
def get_cached_index(index_name_):
    index_ = slse.get_index(index_name_)
    return index_


password = st.text_input("Enter your openai api key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
if password:
    os.environ['OPENAI_API_KEY'] = password

tab2, tab1 = st.tabs(["Use existing data", "Load new data"])
with tab1:
    name = "_".join(st.text_input('Name this index').lower().split())
    uploaded_file = st.file_uploader("Choose a file")
    sp = st.text_input('Split text to sentences by token or split by chunk of chars', value=1000)
    ix = st.button("Index")
    txt = []
    if uploaded_file and sp and name and ix:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file_name = os.path.join(tmpdir, uploaded_file.name)
            with open(tmp_file_name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            txt: List[str] = slse.split_text(tmpdir, sp)
            st.table(txt[0:10])

    if txt:
        with st.spinner('Wait for indexing'):
            slse.get_index(name, txt)
        st.success('Done!')


def clear():
    st.session_state.q = ""


with tab2:
    indexes = os.listdir(slse.storage_dir)
    index_name = st.selectbox(
        'Choose an index to query',
        indexes, key="index_name", on_change=clear)

    st.text_input('Ask a question', key="q")

    if st.session_state.q:
        index = get_cached_index(st.session_state.index_name)
        with st.spinner('Wait for query result'):
            a = slse.query(st.session_state.q, index)
        st.success('Done!')
        st.markdown(a)
        data = []
        for node in a.source_nodes:
            nn = {'text': node.node.text, 'score': node.score}
            data.append(nn)
        st.table(data)
