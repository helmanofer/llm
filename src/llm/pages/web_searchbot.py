import os

import streamlit as st
from langchain import VectorDBQAWithSourcesChain, VectorDBQA
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import DuckDuckGoSearchAPIWrapper

from services.search_store_service import ChromaVectorStore

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k",
    streaming=True,
)

index_ = ChromaVectorStore("cellcom_catalog")
db_vec = VectorDBQA.from_chain_type(llm=llm, vectorstore=index_.vector_store())
st.set_page_config(page_title="LangChain: Chat with search", page_icon="🦜")
st.title("🦜 LangChain: Chat with search")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    if not os.environ.get("OPENAI_API_KEY", ""):
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    search_agent = initialize_agent(
        tools=[
            DuckDuckGoSearchRun(
                name="Search", api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=20),
                description="""Useful for when you need to answer questions about current events or 
                            IMDB/Rotten tomatoes scores. Input should be a search query. 
                            NOT USEFUL to search movies or programs"""
            ),
            Tool(
                name="Vector search",
                func=db_vec.run,
                description="Useful when need to search movies or tv programs",
            ),
        ],
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
