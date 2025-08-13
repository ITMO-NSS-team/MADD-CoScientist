import base64
from io import BytesIO
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from PIL import Image

from ChemCoScientist.paper_analysis.s3_connection import s3_service


def convert_to_base64(file_path):
    """
    Convert PIL images to Base64 encoded strings

    :param file_path: path to image
    :return: Re-sized Base64 string
    """
    if file_path.startswith("http://"):
        s3_key, bucket_name = extract_s3_bucket_and_key(file_path)
        pil_image = Image.open(BytesIO(s3_service.get_image_bytes_from_s3(s3_key, bucket_name)))
    else:
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


def extract_s3_bucket_and_key(s3_url: str):
    o = urlparse(s3_url)
    bucket, key = o.path.split('/', 2)[1:]
    return key, bucket
