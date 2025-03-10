import base64
from PIL import Image
import os


def convert_to_base64(image_file_path):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    pil_image = Image.open(image_file_path)
    pil_image.save('tmp.png', format="png")

    with open('tmp.png', "rb") as image_file:
        result = base64.b64encode(image_file.read()).decode("utf-8")
        os.remove('tmp.png')
        return result
    
def convert_to_html(img_base64):
    """
    Disply base64 encoded string as image

    :param img_base64:  Base64 string
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" style="max-width: 100%;"/>'
    return image_html