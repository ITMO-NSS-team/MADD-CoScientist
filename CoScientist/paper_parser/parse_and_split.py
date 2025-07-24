import logging
import os
from pathlib import Path
import re

from bs4 import BeautifulSoup, Tag
from docling.document_converter import DocumentConverter
from docling_core.types.doc import ImageRefMode
from dotenv import load_dotenv
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered, save_output
from protollm.connectors import create_llm_connector

from definitions import CONFIG_PATH, ROOT_DIR
from CoScientist.paper_parser.parser_prompts import cls_prompt, table_extraction_prompt
from CoScientist.paper_parser.utils import prompt_func, convert_to_base64

_log = logging.getLogger(__name__)

load_dotenv(CONFIG_PATH)

PARSE_RESULTS_PATH = os.path.join(ROOT_DIR, os.environ["PARSE_RESULTS_PATH"])
PAPERS_PATH = os.path.join(ROOT_DIR, os.environ["PAPERS_STORAGE_PATH"])
VISION_LLM_URL = os.environ["VISION_LLM_URL"]
VISION_LLM_NAME = VISION_LLM_URL.split(';')[1]
LLM_SERVICE_CC_URL = os.environ["LLM_SERVICE_CC_URL"]
LLM_SERVICE_KEY = os.getenv("LLM_SERVICE_KEY")
MARKER_LLM = os.getenv("MARKER_LLM")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")
IMAGE_RESOLUTION_SCALE = 2.0
        
        
def simple_conversion(path: str|Path):
    converter = DocumentConverter()
    result = converter.convert(path)
    return result.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)


def parse_with_marker(paper_name: str, use_llm: bool=False) -> (str, Path):
    config = {
        "output_format": "html",
        "use_llm": use_llm,
        "openai_api_key": LLM_SERVICE_KEY,
        "openai_model": MARKER_LLM,
        "openai_base_url": LLM_SERVICE_URL
    }
    config_parser = ConfigParser(config)
    
    file_name = Path(paper_name)
    
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config=config_parser.generate_config_dict(),
        renderer=config_parser.get_renderer(),
        llm_service="marker.services.openai.OpenAIService"
    )
    rendered = converter(paper_name)
    text, _, images = text_from_rendered(rendered)
    
    output_dir = Path(PARSE_RESULTS_PATH, str(file_name.stem) + "_marker")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_output(rendered, output_dir=str(output_dir), fname_base=f"{file_name.stem}")
    return file_name.stem, output_dir


def clean_up_html(paper_name: str, doc_dir: Path) -> str:
    
    file_name = Path(paper_name + ".html")
    parsed_file_path = Path(doc_dir, file_name)
    with open(parsed_file_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    soup = BeautifulSoup(html, "lxml")
    
    blacklist = [
        "author information", "associated content", "acknowledgment", "acknowledgement", "acknowledgments",
        "acknowledgements", "references", "data availability", "declaration of competing interest",
        "credit authorship contribution statement", "funding", "ethical statements", "supplementary materials",
        "conflict of interest", "conflicts of interest", "author contributions", "data availability statement",
        "ethics approval", "supplementary information"
    ]
    for header in soup.find_all(["h1", "h2", "h3"]):
        header_text = header.get_text(strip=True).lower()
        
        if any(exclude in header_text for exclude in blacklist):
            next_node = header.next_sibling
            
            elements_to_remove = []
            while next_node and next_node.name not in ["h1", "h2"]:
                elements_to_remove.append(next_node)
                next_node = next_node.next_sibling
            
            header.decompose()
            for element in elements_to_remove:
                if isinstance(element, Tag):
                    element.decompose()
    
    llm = create_llm_connector(VISION_LLM_URL)
    for img in soup.find_all('img'):
        img_path = str(doc_dir) + "/" + img.get("src")
        images = list(map(convert_to_base64, [img_path]))
        query = [prompt_func({"text": cls_prompt, "image": images})]
        res_1 = llm.invoke(query).content
        if res_1.strip() == "False":
            parent_p = img.find_parent('p')
            if parent_p:
                parent_p.decompose()
                os.remove(img_path)
        else:
            table_query = [prompt_func({"text": table_extraction_prompt, "image": images})]
            res_2 = llm.invoke(table_query).content
            if res_2 != "No table":
                pattern = r'<table\b[^>]*>.*?</table>'
                match = re.search(pattern, res_2, re.DOTALL)
                if match:
                    html_table = match.group(0)
                    table_soup = BeautifulSoup(html_table, 'html.parser')
                    parent_p = img.find_parent('p')
                    if parent_p:
                        parent_p.replace_with(table_soup)

    with open(Path(doc_dir, f"{file_name.stem}_processed.html"), "w", encoding='utf-8') as file:
        file.write(str(soup.prettify()))

    return soup.prettify()
    

def html_chunking(html_string: str, paper_name: str) -> list:
    
    def custom_table_extractor(table_tag):
        return str(table_tag).replace("\n", "")
    
    headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]
    
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=headers_to_split_on,
        max_chunk_size=2500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". "],
        elements_to_preserve=["ul", "table", "ol"],
        preserve_images=True,
        custom_handlers={"table": custom_table_extractor}
    )
    
    documents = splitter.split_text(html_string)
    for doc in documents:
        doc.page_content = "passage: " + doc.page_content  # Maybe delete "passage: " addition
        doc.metadata["imgs_in_chunk"] = str(extract_img_url(doc.page_content, paper_name))
        doc.metadata["source"] = paper_name + ".pdf"
        
    return documents


def extract_img_url(doc_text: str, p_name: str):
    pattern = r'!\[image:([^\]]+\.jpeg)\]\(([^)]+\.jpeg)\)'
    matches = re.findall(pattern, doc_text)
    return [os.path.join(PARSE_RESULTS_PATH, p_name, entry[0]) for entry in matches]


if __name__ == "__main__":
    # documents = loader(parse_and_clean(
    #     "papers/10_1021_acs_joc_0c02350.pdf"))
    # for document in documents:
    #     print(document)
    
    p_path = PAPERS_PATH
    paper = "kowalska-et-al-2023-visible-light-promoted-3-2-cycloaddition-for-the-synthesis-of-cyclopenta-b-chromenocarbonitrile.pdf"
    paper_path = os.path.join(p_path, paper)
    f_name, dir_name = parse_with_marker(paper_name=paper_path)
    parsed_paper = clean_up_html(paper_name=f_name, doc_dir=dir_name)
    chunks = html_chunking(html_string=parsed_paper, paper_name=f_name)
    print(chunks)
