import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pathlib import Path

from chroma_db_operations import store_images_in_chromadb_txt_format
from pipelines import get_or_create_chroma_collection, process_question, upload_paper, prepare_db

logger = st.logger.get_logger(__name__)

load_dotenv('../config.env')

logger.info(str(Path(__file__).resolve().parent.parent))

BASE_LLM_URL = 'https://api.vsegpt.ru/v1;vis-google/gemini-2.0-flash-001'
SUM_LLM_URL = 'https://api.vsegpt.ru/v1;vis-google/gemini-2.0-flash-001'

SUM_COLLECTION_NAME = 'paper_summaries_img2txt'
TXT_COLLECTION_NAME = 'text_context_img2txt'
IMG_COLLECTION_NAME = 'image_context'

SUM_CHUNK_NUM = 10
TXT_CHUNK_NUM = 6
IMG_CHUNK_NUM = 6

PAPER_STORAGE = '../papers'


def _get_collections():
    sum_collection = get_or_create_chroma_collection(SUM_COLLECTION_NAME)
    txt_collection = get_or_create_chroma_collection(TXT_COLLECTION_NAME)
    img_collection = get_or_create_chroma_collection(IMG_COLLECTION_NAME)

    return sum_collection, txt_collection, img_collection


def get_answer_from_assistant(question: str) -> tuple[str, list, str, dict]:
    logger.info('Retrieving answer from assistant...')

    sum_collection, txt_collection, img_collection = _get_collections()

    logger.info('Connected to ChromaDB collections')

    return process_question(sum_collection, txt_collection, img_collection,
                            SUM_CHUNK_NUM, TXT_CHUNK_NUM, IMG_CHUNK_NUM,
                            question, BASE_LLM_URL)


def process_uploaded_file(uploaded_file):
    res = {'success': False, 'msg': ''}

    sum_collection, txt_collection, img_collection = _get_collections()

    uploaded_file_path = Path(PAPER_STORAGE, uploaded_file.name)
    logger.info('preped all vars')
    if not uploaded_file_path.exists():
        logger.info('file does not exist')
        try:
            with open(uploaded_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            logger.info('file uploaded')
            upload_paper(uploaded_file_path, sum_collection, txt_collection,
                         img_collection, store_images_in_chromadb_txt_format, SUM_LLM_URL)
            logger.info('file added to db')
            res['success'] = True
            res['msg'] = 'Successfully added paper to DB!'
        except Exception as e:
            Path(uploaded_file_path).unlink(missing_ok=True)
            logger.error(f'Could not process file: {e}')
            res['msg'] = 'Could not process file!'
    else:
        logger.info('file already exists')
        res['success'] = True
        res['msg'] = 'File already exists!'
    logger.info('finished processing')
    return res


def load_papers():
    logger.info('Prepping DB...')
    reg_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-small",
        normalize_embeddings=True
    )
    logger.info('Loaded embedding function')
    prepare_db(SUM_COLLECTION_NAME, TXT_COLLECTION_NAME, IMG_COLLECTION_NAME,
               reg_embedding_function, reg_embedding_function, reg_embedding_function,
               store_images_in_chromadb_txt_format, PAPER_STORAGE, BASE_LLM_URL)
    logger.info('DB ready!')


if __name__ == "__main__":
    # question = 'how does the synthesis of Glionitrin A/B happen?'
    # print(get_answer_from_assistant(question))
    load_papers()
