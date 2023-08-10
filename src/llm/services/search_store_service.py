import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Any

import openai
import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma, VectorStore

from common.models import QuestionAnswer, DocumentWithScore

openai.log = "info"
MAX_TOKENS = 8192

# logging.basicConfig()
# logging.getLogger("langchain").setLevel(logging.INFO)


class BaseSearchService:
    version = "v0.1"

    def __init__(self, index_name, api_key: str = None):
        self.index_name = index_name
        self.index_dir = Path(self.storage_dir()).joinpath(self.index_name).as_posix()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            openai.api_key = api_key

    @property
    def embeddings(self):
        return OpenAIEmbeddings()

    @staticmethod
    def storage_dir():
        return Path(__file__).resolve().parent.parent.joinpath("storage").as_posix()

    @classmethod
    def list_indexes(cls):
        cls.storage_dir()
        try:
            return os.listdir(cls.storage_dir())
        except FileNotFoundError:
            return []

    @staticmethod
    def chunk_text(text, chunk_size):
        overlap_size = 100
        chunks = []
        for i in range(0, len(text), chunk_size - overlap_size):
            chunks.append(text[i : i + chunk_size])
        return chunks

    def split_text(self, path: str, chunk_by: str | int) -> List[str]:
        from llama_index import SimpleDirectoryReader

        chunk = None
        split_token = None
        try:
            chunk = int(chunk_by)
        except ValueError:
            split_token = chunk_by

        documents = SimpleDirectoryReader(path).load_data()
        txt_l: List[str] = []
        for doc in documents:
            txt = doc.text
            if chunk:
                txt_l += self.chunk_text(txt, chunk)
            else:
                txt_l += txt.split(split_token)
        return txt_l

    @lru_cache
    def query(self, query: str) -> QuestionAnswer:
        from langchain.chains.question_answering import load_qa_chain
        from langchain.chat_models import ChatOpenAI

        vec_store = self.vector_store()
        text = query.strip()
        chat = ChatOpenAI(model_name="gpt-4")

        chain = load_qa_chain(chat, chain_type="stuff", verbose=True)
        unique_docs_with_scores = vec_store.similarity_search_with_score(text, k=20)
        enc = tiktoken.get_encoding("cl100k_base")
        cnt = 0
        docs_w_scored = []
        for doc, score in unique_docs_with_scores:
            cnt += len(enc.encode(doc.page_content))
            print(cnt)
            if cnt >= MAX_TOKENS:
                print("break")
                break
            docs_w_scored.append(DocumentWithScore(text=doc.page_content, score=score))

        unique_docs, scores = zip(*unique_docs_with_scores)
        r = chain(
            {"input_documents": unique_docs, "question": text},
            return_only_outputs=True,
        )
        response_text = r["output_text"]
        return QuestionAnswer(answer=response_text, documents=docs_w_scored)

    def vector_store(self) -> VectorStore:
        raise NotImplementedError

    def store_documents(self, docs: List[str]):
        raise NotImplementedError


class ChromaVectorStore(BaseSearchService):
    def vector_store(self) -> Chroma:
        db = Chroma(
            persist_directory=self.index_dir,
            collection_name=self.index_name,
            embedding_function=self.embeddings,
        )
        return db

    def store_documents(self, docs: List[str]):
        chunk = 200
        for i in range(0, len(docs), chunk):
            cdocs = [Document(page_content=d) for d in docs[i : i + chunk]]
            store = self.vector_store()
            store.add_documents(cdocs)
            store.persist()


from langchain.agents import tool


@tool()
def search_catalog(query: str) -> List[str]:
    """Useful when need to search movie or tv program catalog"""
    a = ChromaVectorStore("tvm")
    res = a.query(query)
    full_res = [r.text for r in res.documents] + [res.answer]
    return [res.answer]
