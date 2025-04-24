import os
from chromadb.api.models import Collection
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from typing import Callable

from answer_question import query_llm
from chroma_db_operations import get_or_create_chroma_collection, \
    store_mm_embeddings_in_chromadb, query_chromadb, store_txt_embeddings_in_chromadb
from parse_and_split import loader, parse_and_clean


IMAGES_PATH = './parse_results'
load_dotenv('../config.env')


def process_questions(questions_path: str,
                      answers_path: str,
                      chroma_collection: Collection,
                      llm_url: str) -> None:
    df = pd.read_csv(questions_path)
    for index, row in df.iterrows():
        question = row.question
        context = query_chromadb(chroma_collection, question)
        txt_context = ''
        img_paths = []

        for ind, chunk in enumerate(context['metadatas'][0]):
            if chunk['type'] == 'image':
                img_paths.append(chunk['image_path'])
            if chunk['type'] == 'text':
                txt_context += '\n\n' + context['documents'][0][ind]

        ans, metadata = query_llm(llm_url, question, txt_context, img_paths)

        df.at[index, 'chroma_text_context'] = txt_context
        df.at[index, 'chroma_images_context'] = str(img_paths)
        df.at[index, 'llm_answer'] = ans

    df.to_csv(answers_path, index=False)


def load_data_to_chroma(papers_path: str,
                        chroma_collection: Collection,
                        store_function: Callable) -> None:
    for paper in os.listdir(papers_path):
        paper_path = os.path.join(papers_path, paper)
        parsed_images_path = os.path.join(IMAGES_PATH, Path(paper).stem)

        documents = loader(parse_and_clean(paper_path))
        store_function(chroma_collection, documents, parsed_images_path, paper)


def run_mm_rag():
    collection_name = 'mm_data'
    embedding_function = embedding_functions.OpenCLIPEmbeddingFunction()

    llm_url = 'https://api.vsegpt.ru/v1;vis-google/gemini-2.0-flash-001'

    papers_path = './papers'
    questions_path = './questions/simple_questions.csv'
    answers_path = './questions/simple_mm_answers.csv'

    chroma_collection = get_or_create_chroma_collection(collection_name, embedding_function)

    load_data_to_chroma(papers_path, chroma_collection, store_mm_embeddings_in_chromadb)

    process_questions(questions_path, answers_path, chroma_collection, llm_url)


def run_img2txt_rag():
    collection_name = "text_data"

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large",
        normalize_embeddings=True
    )

    chroma_collection = get_or_create_chroma_collection(collection_name, embedding_function)

    llm_url = 'https://api.vsegpt.ru/v1;vis-google/gemini-2.0-flash-001'

    papers_path = './papers'
    questions_path = './questions/simple_questions.csv'
    answers_path = './questions/simple_text_answers.csv'

    load_data_to_chroma(papers_path, chroma_collection, store_txt_embeddings_in_chromadb)

    process_questions(questions_path, answers_path, chroma_collection, llm_url)


if __name__ == "__main__":
    # run_mm_rag()
    run_img2txt_rag()
