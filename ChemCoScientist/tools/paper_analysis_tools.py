import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools.render import render_text_description
from langchain_core.tools import tool
from protollm.connectors import create_llm_connector
from pathlib import Path

from ChemCoScientist.paper_analysis.chroma_db_operations import ChromaDBPaperStore
from ChemCoScientist.paper_analysis.prompts import paraphrase_prompt
from ChemCoScientist.paper_analysis.question_processing import process_question, simple_query_llm
from ChemCoScientist.frontend.memory import SELECTED_PAPERS
from definitions import CONFIG_PATH

load_dotenv(CONFIG_PATH)
VISION_LLM_URL = os.environ["VISION_LLM_URL"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def explore_chemistry_database(task: str) -> dict:
    """Answers questions by retrieving and analyzing information from a database
    of chemical scientific papers. Using this tool takes precedence over web search.
    This tool is ONLY for answering general questions, it cannot be used for questions about
    specific papers (e.g. 'what is on Figure 1?', 'what experiments are described?',
    'what are the conclusions?', etc). It cannot be used to find a list of relevant papers.

    Args:
        task (str): user query for the DB with chemical papers

    Returns:
        A dictionary with the final response from the LLM and metadata:
        the text, images and tables that were used as context for the request,
        metadata of the request (model, number of used tokens, etc)
    """
    try:
        print('Running explore_chemistry_database tool...')
        print(f'task: {task}')
        return process_question(task)
    except Exception as e:
        logger.error(f'explore_chemistry_database ERROR: {e}')
        return {'answer': 'Could not extract any data from DB.'}


@tool
def explore_my_papers(task: str, session_id: str = None) -> dict:
    """Answers questions on chemistry using specific scientific papers that were uploaded
    by the user. Using this tool takes precedence over web search.
    This tool is ONLY for questions about specific papers (e.g. 'what is on Figure 1?',
    'what experiments are described?', 'what are the conclusions?', etc).
    The tool cannot be used for general questions or to find relevant papers.

    Args:
        task (str): user query for the user's papers
        session_id (str): current session ID, required only for frontend

    Returns:
        A dictionary with the final response from the LLM and metadata
        of the request (model, number of used tokens, etc)
    """
    print('Running explore_my_papers tool...')
    print(f'task: {task}')
    try:
        # TODO: remove when proper frontend is added
        if not SELECTED_PAPERS:
            directory = Path(os.environ.get('MY_PAPERS_PATH'))
            papers = [str(f.resolve()) for f in directory.iterdir() if f.is_file() and f.suffix.lower() == '.pdf']

            if not papers:
                return {'answer': 'No papers provided for search.'}
        else:
            if not SELECTED_PAPERS.get(session_id, []):
                return {'answer': 'No papers provided for search.'}
            papers = SELECTED_PAPERS[session_id]

        return simple_query_llm(VISION_LLM_URL, task, papers)
    except Exception as e:
        logger.error(f'explore_my_papers ERROR: {e}')
        return {'answer': 'Could not extract any data from uploaded papers.'}


def paraphrase_query(query: str) -> dict:
    """
    Converts the user's initial question about chemistry into a more suitable
    query for semantic search.

    Args:
        query: user's query

    Returns:
        A new query for a better search for suitable papers.
    """
    llm = create_llm_connector(VISION_LLM_URL)

    user_message = {"type": "text", "text": f"USER QUESTION: {query}"}

    messages = [
        SystemMessage(content=paraphrase_prompt),
        HumanMessage(content=user_message),
    ]

    res = llm.invoke(messages)
    return {'answer': res.content}


@tool
def select_papers(query: str, papers_num: int = 15, final_papers_num: int = 3) -> dict:
    """
    Finds the specified number of papers for the user's request based on a database with
    chemical scientific papers. Using this tool takes precedence over web search.
    Must be used ONLY when user requests to list papers and must NOT be used to answer questions.


    Args:
        query: user's query
        initial_papers_num: number of papers to search through the vector storage before re-ranking
        papers_number_after_reranking: the number of papers that will remain after re-ranking

    Returns:
        A list of papers suitable for the user's request
    """
    try:
        print('Running select_papers tool...')
        print(f'query: {query}')
        paper_store = ChromaDBPaperStore()
        return paper_store.search_for_papers(query, papers_num, final_papers_num)
    except Exception as e:
        logger.error(f'select_papers ERROR: {e}')
        return {'answer': 'Could not find any papers in DB.'}


paper_analysis_tools = [
    explore_chemistry_database,
    explore_my_papers,
    select_papers,
]

paper_analysis_tools_rendered = render_text_description(paper_analysis_tools)
