import base64
from io import BytesIO

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from protollm.connectors import create_llm_connector

load_dotenv("config.env")

sys_prompt = ("You are a helpful chemist assistant. Answer USER QUESTION in a direct tone. Be"
              " moderately concise. Your audience is an expert, so be highly specific. If there are"
              " ambiguous terms or acronyms, first define them. For answer you must use CONTEXT"
              " provided by user. Just answer the question."
              "\nRules:\n1. You must use only provided information for the answer.\n2. Add a unit of"
              " measurement to an answer only if appropriate.\n3. For answer you should take only that"
              " information from context, which is relevant to user's question.\n4. If you do not know"
              " how to answer the questions, say so.\n5. If you are additionally given images, you can"
              " use the information from them as CONTEXT to answer.\n 6. Use valid IUPAC or SMILES "
              " notation if necessary to answer the question.")


def convert_to_base64(file_path):
    """
    Convert PIL images to Base64 encoded strings

    :param file_path: path to image
    :return: Re-sized Base64 string
    """
    pil_image = Image.open(file_path)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def prompt_func(data):
    text = data["text"]
    imgs = data["image"]
    content_parts = []
    
    for img in imgs:
        image_part = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{img}",
        }
        content_parts.append(image_part)
    
    text_part = {"type": "text", "text": text}
    content_parts.append(text_part)
    
    return HumanMessage(content=content_parts)

file_paths = []  # Enter list of paths to images here

images = list(map(convert_to_base64, file_paths))

llm = create_llm_connector("https://api.vsegpt.ru/v1;vis-google/gemini-2.0-flash-001")

if __name__ == "__main__":
    # question = ("Какая реакция идет протекает на 6 стадии Total Synthesis of (−)-Glionitrin A/B? Какие реагенты"
    #             " участвовали в реакции и какой продукт получили? Какой получился выход?")
    question = ("I need all the compounds that were used in the experiments. Obligatorily I need all results to be in"
                " the form of a table of 2 columns where in the first column were the names by IUPAC numberclature and"
                " in the second column in SMILES notation. Don't add it to this list of reaction products for me. Can"
                " you do that?")
    context = ""
    
    messages = [
        SystemMessage(content=sys_prompt),
        prompt_func({"text": f"USER QUESTION: {question}\n\nCONTEXT: {context}", "image": images})
    ]
    # messages = [
    #     SystemMessage(content="You're a useful assistant. You only ever reply in the form of valid JSON."),
    #     prompt_func(
    #         {
    #             "text": "For the provided images, generate a detailed clear description. If there is a table in the"
    #                     " image, parse it and return it in HTML format. If you see chemical compounds in the figures,"
    #                     " output the names of the compounds according to IUPAC nomenclature.\n"
    #                     " As a response, return ONLY JSON of the following form: {‘figure_1’:"
    #                     " ‘figure_1_description’, ‘figure_2’: ‘figure_2_description’, ‘table_1’:"
    #                     " ‘table_1_description’...}",
    #             "image": images
    #         }
    #     )
    # ]
    
    res = llm.invoke(messages)
    print(res.content)
    print(res.response_metadata)
