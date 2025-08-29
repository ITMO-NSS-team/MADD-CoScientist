import base64
import json
import os
from typing import Tuple

import pandas as pd
import requests
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


def get_all_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


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


def generate_for_base_case(
    numb_mol: int = 1,
    cuda: bool = True,
    mean_: float = 0.0,
    std_: float = 1.0,
    url: str = f"http://{os.environ.get('MODEL_API_ADDR_BASE_CASE')}/case_generator",
    case_: str = "RNDM",
    **kwargs,
) -> Tuple[requests.models.Response, dict]:
    """Function that call Chem server API for generate molecules with properties by choosen case. By default it call random generation case.

    Args:
        numb_mol (int, optional): Number of moluecules that need to generate. Defaults to 1.
        cuda (bool, optional): Cuda usage mode. Defaults to True.
        mean_ (float, optional): mean of noise distibution. ONLY FOR EXPERIMENTS. Defaults to 0.0.
        std_ (float, optional): std of noise distibution. ONLY FOR EXPERIMENTS. Defaults to 1.0.
        url (_type_, optional): URL to API srver. Defaults to 'http://10.32.2.4:80/case_generator'.
        case_ (str, optional): Key for api, that define what case you choose for. Can be choose from: 'Alzhmr','Sklrz','Prkns','Cnsr','Dslpdm','TBLET', 'RNDM'.
                               Where: 'Alzhmr' - Alzheimer,
                                        'Sklrz' - Skleroz,
                                        'Prkns' - Parkinson,
                                        'Cnsr' - Canser,
                                        'Dslpdm' - Dyslipidemia,
                                        'TBLET' - Drug resistance,
                                        'RNDM' - random generation.
                                        Defaults to RNDM.
    Returns:
        Tuple[requests.models.Response, dict]: Return full respones, or just dict with molecules and properties list.
        Tuple[requests.models.Response, dict]: Return full respones, or just dict with molecules and properties list.

    Example:
        numbs = 4
        params = {'numb_mol': numbs, 'cuda': False, 'mean_': 0, case_ = 'RNDM
                'std_': 1}
        resp_mol, mols = call_for_generation(**params,hello='world')
        print(mols)
        >> {'Molecules': ['Cc1cc(C(=O)OCC(=O)NCC2CCCO2)nn1C', 'CSC1=CC=C(C(=O)O)C(C(=O)c2ccc(C(F)(F)F)cc2)S1', 'CSc1cc(-c2ccc(-c3ccccc3)cc2)nc(C(C)=O)c1O', 'CC(C)N(CC(=O)NCc1cn[nH]c1)Cc1ccccc1'],
          'Docking score': [-6.707, -7.517, -8.541, -7.47],
            'QED': [0.7785404162969669, 0.8150693008303525, 0.5355361484098266, 0.8174264075095671],
              'SA': [2.731063371805302, 3.558887012627684, 2.2174895913203354, 2.2083851588937087],
                'PAINS': [0, 0, 0, 0],
                  'SureChEMBL': [0, 0, 0, 0],
                    'Glaxo': [0, 0, 0, 0]}
    """

    params = {
        "numb_mol": numb_mol,
        "cuda": cuda,
        "mean_": mean_,
        "std_": std_,
        "case_": case_,
        **kwargs,
    }
    try:
        resp = requests.post(url, data=json.dumps(params))
        if resp.status_code != 200:
            print(
                "ERROR: response status code from requests to generative model: ",
                resp.status_code,
            )

    except requests.exceptions.RequestException as e:
        print(e)

    return resp, json.loads(resp.json())
