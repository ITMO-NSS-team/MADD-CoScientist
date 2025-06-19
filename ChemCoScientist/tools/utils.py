import base64
import os

from PIL import Image


def convert_to_base64(image_file_path):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    pil_image = Image.open(image_file_path)
    pil_image.save("tmp.png", format="png")

    with open("tmp.png", "rb") as image_file:
        result = base64.b64encode(image_file.read()).decode("utf-8")
        os.remove("tmp.png")
        return result


def convert_to_html(img_base64):
    """
    Disply base64 encoded string as image

    :param img_base64:  Base64 string
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = (
        f'<img src="data:image/jpeg;base64,{img_base64}" style="max-width: 100%;"/>'
    )
    return image_html


import pandas as pd


def filter_valid_strings(
    df: pd.DataFrame, column_name: str, max_length: int = 200
) -> pd.DataFrame:
    """
    Removes molecules longer than 200 characters.

    Example:
    -------
    >>> df = pd.DataFrame({'text': ['abc', 'def'*100, 123]})
    >>> filtered_df = filter_valid_strings(df, 'text')
    >>> print(filtered_df)
    """
    try:
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found")

        is_string = df[column_name].apply(lambda x: isinstance(x, str))
        valid_length = df[column_name].str.len() <= max_length

        filtered_df = df[is_string & valid_length].copy()

        return filtered_df

    except Exception as e:
        raise ValueError(e)
