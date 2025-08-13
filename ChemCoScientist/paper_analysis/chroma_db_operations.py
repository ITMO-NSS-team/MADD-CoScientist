import base64
import logging
import os
import uuid
from pathlib import Path

import chromadb
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.models import Collection
from chromadb.utils.data_loaders import ImageLoader
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage
from protollm.connectors import create_llm_connector
from pydantic import BaseModel, Field
import requests

from ChemCoScientist.paper_analysis.prompts import summarisation_prompt
from ChemCoScientist.paper_analysis.settings import allowed_providers
from ChemCoScientist.paper_analysis.settings import settings as default_settings
from ChemCoScientist.paper_analysis.s3_connection import s3_service
from CoScientist.paper_parser.parse_and_split import (
    clean_up_html,
    html_chunking,
)
from definitions import CONFIG_PATH, ROOT_DIR

load_dotenv(CONFIG_PATH)
DATA_LOADER = ImageLoader()
IMAGES_PATH = os.path.join(ROOT_DIR, os.environ["PARSE_RESULTS_PATH"])
CHROMA_DB_PATH = os.path.join(ROOT_DIR, os.environ["CHROMA_STORAGE_PATH"])
VISION_LLM_URL = os.environ["VISION_LLM_URL"]
SUMMARY_LLM_URL = os.environ["SUMMARY_LLM_URL"]
PAPERS_PATH = os.path.join(ROOT_DIR, os.environ["PAPERS_STORAGE_PATH"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpandedSummary(BaseModel):
    """Expanded version of paper's summary."""
    paper_summary: str = Field(description="Summary of the paper.")
    paper_title: str = Field(
        description="Title of the paper. If the title is not explicitly specified, use the default value - 'NO TITLE'"
    )
    publication_year: int = Field(
        description=(
            "Year of publication of the paper. If the publication year is not explicitly specified, use the default"
            " value - 9999."
        )
    )


class CustomEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        embeddings = ChromaDBPaperStore.get_embeddings(texts)
        return embeddings


class ChromaClient:
    def __init__(self):
        # self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.client = chromadb.HttpClient(
            host=default_settings.chroma_host,
            port=default_settings.chroma_port,
            settings=chromadb.Settings(allow_reset=default_settings.allow_reset),
        )

    def get_or_create_chroma_collection(
        self,
        collection: str,
        embedding_function: EmbeddingFunction[Documents] | None = None,
    ) -> Collection:
        return self.client.get_or_create_collection(
            name=collection,
            embedding_function=embedding_function,
            data_loader=DATA_LOADER,
        )

    @staticmethod
    def query_chromadb(
        collection: Collection,
        query_text: str,
        metadata_filter: dict = None,
        chunk_num: int = 3,
    ) -> dict:
        return collection.query(
            query_texts=[query_text],
            n_results=chunk_num,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )
    
    def delete_collection(self, name: str):
        self.client.delete_collection(name)
        
    def show_collections(self):
        return self.client.list_collections()


class ChromaDBPaperStore:
    def __init__(self):
        self.llm_url = VISION_LLM_URL

        self.client = ChromaClient()

        self.sum_collection_name = os.getenv("SUMMARIES_COLLECTION_NAME")
        self.txt_collection_name = os.getenv("TEXTS_COLLECTION_NAME")
        self.img_collection_name = os.getenv("IMAGES_COLLECTION_NAME")

        self.sum_chunk_num = 15
        self.final_sum_chunk_num = 3
        self.txt_chunk_num = 15
        self.img_chunk_num = 2

        self.sum_collection = self.client.get_or_create_chroma_collection(
            self.sum_collection_name, CustomEmbeddingFunction()
        )
        self.txt_collection = self.client.get_or_create_chroma_collection(
            self.txt_collection_name, CustomEmbeddingFunction()
        )
        self.img_collection = self.client.get_or_create_chroma_collection(
            self.img_collection_name, CustomEmbeddingFunction()
        )
        self.workers = 2

    @staticmethod
    def _image_to_base64(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_string

    def _image_to_text(self, image_path: str) -> str:
        sys_prompt = (
            "This is an image from a scientific paper in chemistry. "
            "Write a short but succinct description of the image that reflects its essence."
            "Be as concise as possible. "
            "Only use data from image, do NOT make anything up."
        )
        model = create_llm_connector(
            self.llm_url, temperature=0.015, top_p=0.95, extra_body={"provider": {"only": allowed_providers}}
        )
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": sys_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._image_to_base64(image_path)}"
                        },
                    },
                ],
            )
        ]
        res = model.invoke(messages)
        return res.content

    def store_text_chunks_in_chromadb(self, content: list) -> None:
        embeddings = self.get_embeddings([text_chunk.page_content for text_chunk in content])
        self.txt_collection.add(
            ids=[str(uuid.uuid4()) for _ in range(len(content))],
            documents=[text_chunk.page_content for text_chunk in content],
            embeddings=embeddings,
            metadatas=[{"type": "text", **text_chunk.metadata} for text_chunk in content]
        )

    def store_images_in_chromadb_txt_format(self, image_dir: str, paper_name: str, url_mapping: dict) -> None:
        image_descriptions = []
        image_paths = []
        image_counter = 0
        valid_paths = list(url_mapping.keys())

        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(image_dir, filename)
                if img_path in valid_paths:
                    image_descriptions.append(self._image_to_text(img_path))
                    image_paths.append(url_mapping[img_path])
                    image_counter += 1

        embeddings = self.get_embeddings(image_descriptions)
        self.img_collection.add(
            ids=[str(uuid.uuid4()) for _ in range(image_counter)],
            documents=image_descriptions,
            embeddings=embeddings,
            metadatas=[
                {"type": "image", "source": paper_name, "image_path": img_path} for img_path in image_paths
            ]
        )


    def search_for_papers(self,
                          query: str,
                          chunks_num: int = None,
                          final_chunks_num: int = None) -> dict:
        chunks_num = chunks_num if chunks_num else self.sum_chunk_num
        final_chunks_num = final_chunks_num if final_chunks_num else self.final_sum_chunk_num

        raw_docs = self.client.query_chromadb(self.sum_collection, query, chunk_num=chunks_num)
        docs = self.search_with_reranker(query, raw_docs, top_k=final_chunks_num)
        res = [doc[2]["source"] for doc in docs]
        return {'answer': res}

    def retrieve_context(
            self, query: str, relevant_papers: list = None
    ) -> tuple[list, dict]:
        if not relevant_papers:
            relevant_papers = self.search_for_papers(query)

        raw_text_context = self.client.query_chromadb(
            self.txt_collection,
            query,
            {"source": {"$in": relevant_papers['answer']}},
            self.txt_chunk_num,
        )
        image_context = self.client.query_chromadb(
            self.img_collection,
            query,
            {"source": {"$in": relevant_papers['answer']}},
            self.img_chunk_num,
        )
        text_context = self.search_with_reranker(query, raw_text_context, top_k=5)
        return text_context, image_context
    
    def search_with_reranker(self, query: str, initial_results, top_k: int = 1) -> list[tuple[str, str, dict, float]]:
        metadatas = initial_results['metadatas'][0]
        documents = initial_results["documents"][0]
        ids = initial_results["ids"][0]

        pairs = [[query, doc.replace("passage: ", "")] for doc in documents]
        
        rerank_scores = self.rerank(pairs)

        scored_docs = list(zip(ids, documents, metadatas, rerank_scores))
        scored_docs.sort(key=lambda x: x[3], reverse=True)

        return scored_docs[:top_k]

    def add_paper_summary_to_db(self, paper_name: str, parsed_paper: str, llm) -> None:
        expanded_summary: ExpandedSummary = llm.invoke([HumanMessage(content=summarisation_prompt + parsed_paper)])
        doc = Document(
            page_content=expanded_summary.paper_summary,
            metadata={
                "source": paper_name,
                "paper_title": expanded_summary.paper_title,
                "publication_year": expanded_summary.publication_year
            }
        )
        embedding = self.get_embeddings([doc.page_content])
        self.sum_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[doc.page_content],
            embeddings=embedding,
            metadatas=[{"type": "text", **doc.metadata}]
        )
        print(f"Summary loaded for: {paper_name}")

    def run_marker_pdf(self, p_path, out_path) -> None:
        try:
            os.system(
                " ".join(
                    [
                        "sh",
                        os.path.join(ROOT_DIR, "ChemCoScientist/paper_analysis/marker_parsing.sh"),
                        str(p_path),
                        str(out_path),
                        str(self.workers)
                    ]
                )
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    @staticmethod
    def get_embeddings(texts: list[str]) -> list[np.ndarray]:
        embedding_service_url = "http://" + default_settings.embedding_host + ":"\
                                + str(default_settings.embedding_port)\
                                + default_settings.embedding_endpoint
        try:
            response = requests.post(
                embedding_service_url,
                json=texts,
                timeout=300
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            logger.error(f"Embedding service error: {str(e)}")
            raise
    
    @staticmethod
    def rerank(pairs: list[list[str]]) -> list[float]:
        reranker_service_url = "http://" + default_settings.reranker_host + ":" \
                                + str(default_settings.reranker_port) \
                                + default_settings.reranker_endpoint
        try:
            response = requests.post(
                reranker_service_url,
                json=pairs,
                timeout=300
            )
            response.raise_for_status()
            return response.json()["scores"]
        except Exception as e:
            logger.error(f"Reranker service error: {str(e)}")
            raise


process_local_store: ChromaDBPaperStore = None


def init_process():
    global process_local_store
    process_local_store = ChromaDBPaperStore()


def process_single_document(folder_path: Path):
    paper_name = folder_path.name.replace("_marker", "")
    file_name = Path(paper_name + ".html")
    paper_name_to_load = Path(paper_name + ".pdf")
    parsed_file_path = Path(folder_path, file_name)
    with open(parsed_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    try:
        print(f"Starting post-processing paper: {paper_name}")
        parsed_paper, mapping = clean_up_html(folder_path, folder_path, text, s3_service, paper_name)
        print(f"Finished post-processing paper: {paper_name}")
        documents = html_chunking(parsed_paper, paper_name)
        
        llm = create_llm_connector(SUMMARY_LLM_URL, extra_body={"provider": {"only": allowed_providers}})
        struct_llm = llm.with_structured_output(schema=ExpandedSummary)
        
        print(f"Starting loading paper: {paper_name}")
        process_local_store.add_paper_summary_to_db(str(paper_name_to_load), parsed_paper, struct_llm)
        process_local_store.store_text_chunks_in_chromadb(documents)
        process_local_store.store_images_in_chromadb_txt_format(str(folder_path), str(paper_name_to_load), mapping)
        print(f"Finished loading paper: {paper_name}")
    except Exception as e:
        print(f"Error in {paper_name}: {str(e)}")


def process_all_documents(base_dir: Path):
    folders = [d for d in base_dir.iterdir() if d.is_dir()]
    with ThreadPoolExecutor(max_workers=2, initializer=init_process()) as pool:
        pool.map(process_single_document, [folder for folder in folders])


if __name__ == "__main__":

    p_path = PAPERS_PATH
    res_path = IMAGES_PATH
    
    p_store = ChromaDBPaperStore()
    p_store.run_marker_pdf(p_path, res_path)
    del p_store
    process_all_documents(Path(res_path))
    
    # p_store.client.delete_collection(name="test_paper_summaries_img2txt")
    # p_store.client.delete_collection(name="test_text_context_img2txt")
    # p_store.client.delete_collection(name="test_image_context")
    # print(p_store.client.show_collections())
    
    # print(ChromaDBPaperStore.get_embeddings(["hello", "world"]))
    # print(ChromaDBPaperStore.rerank([["hello", "world"], ["hello", "there"]]))
