import logging
import os
import sys
from pathlib import Path
from typing import List

from llama_index import GPTVectorStoreIndex, Document, StorageContext, load_index_from_storage, SimpleDirectoryReader
from llama_index.indices.base import BaseGPTIndex
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.node_parser import SimpleNodeParser
from llama_index.query_engine import RetrieverQueryEngine

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class SLSE:
    def __init__(self, dir_path: str = None, api_key: str = None):
        self.dir_path = dir_path
        if not dir_path:
            self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.storage_dir = Path(self.dir_path).joinpath("storage")
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    @staticmethod
    def summarize(txt):
        import openai
        q = f"""{txt}

        Tl;dr
        """
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=q,
            temperature=0,
            max_tokens=1500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["###"]
        )

        tldr_txt = response['choices'][0]['text']
        return tldr_txt

    @staticmethod
    def chunk_text(text, chunk_size):
        overlap_size = 100
        chunks = []
        for i in range(0, len(text), chunk_size - overlap_size):
            chunks.append(text[i:i + chunk_size])
        return chunks

    def split_text(self, path: str, chunk_by: str) -> List[str]:
        chunk = None
        split_token = None
        try:
            chunk = int(chunk_by)
        except ValueError:
            split_token = chunk_by

        documents = SimpleDirectoryReader(path).load_data()
        txt = documents[0].text
        if chunk:
            txt_l: List[str] = self.chunk_text(txt, chunk)
        else:
            txt_l: List[str] = txt.split(split_token)
        return txt_l

    def get_index(self, index_name: str, txt: list[str] = None):
        index_dir = f'{self.storage_dir}/{index_name}'
        from llama_index.vector_stores import LanceDBVectorStore
        vector_store = LanceDBVectorStore(uri=index_dir)

        if txt:
            documents = [Document(t) for t in txt if len(t) > 100]
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
            )
            index: GPTVectorStoreIndex = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context)
            index.storage_context.persist(persist_dir=index_dir)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=index_dir, vector_store=vector_store)
            index: BaseGPTIndex = load_index_from_storage(storage_context)
        return index

    def get_index_define_llm(self, index_name: str, txt: list[str] = None):
        from llama_index import LLMPredictor
        from langchain import OpenAI
        from llama_index import ServiceContext

        index_dir = f'{self.storage_dir}/{index_name}'
        from llama_index.vector_stores import LanceDBVectorStore
        vector_store = LanceDBVectorStore(uri=index_dir)
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.3, model_name="davinci", best_of=3))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

        if txt:
            documents = [Document(t) for t in txt if len(t) > 100]
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
            )
            index: GPTVectorStoreIndex = (GPTVectorStoreIndex
                                          .from_documents(documents,
                                                          storage_context=storage_context,
                                                          service_context=service_context)
                                          )
            index.storage_context.persist(persist_dir=index_dir)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=index_dir, vector_store=vector_store)
            index: BaseGPTIndex = load_index_from_storage(storage_context, service_context=service_context)
            print(index.service_context.llm_predictor)
        return index

    def _get_index_simple(self, index_name, txt: list[str] = None):
        index_dir = f'{self.storage_dir}/{index_name}'
        index_file = f'{index_dir}/index_store.json'

        if not Path(index_file).is_file() and txt:
            documents = [Document(t) for t in txt if len(t) > 100]
            parser = SimpleNodeParser()
            nodes = parser.get_nodes_from_documents(documents)
            index: GPTVectorStoreIndex = GPTVectorStoreIndex(nodes)
            index.storage_context.persist(persist_dir=index_dir)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=index_dir)
            index: BaseGPTIndex = load_index_from_storage(storage_context)
        return index

    @staticmethod
    def query(q: str, index: GPTVectorStoreIndex):
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10,
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
        )

        response = query_engine.query(q)
        return response
