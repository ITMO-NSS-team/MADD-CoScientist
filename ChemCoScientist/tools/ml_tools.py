import json
import multiprocessing as mp
import os
import socket
import time
from multiprocessing import Process
from typing import List, Tuple

import pandas as pd
import requests
from langchain.tools.render import render_text_description
from langchain_core.tools import tool

# TODO: make yaml conf
conf = {"url": "http://10.64.4.243:81"}


@tool
def predict_prop_by_smiles(
    smiles_list: List[str], case: str = "test_api", timeout: int = 20
) -> Tuple[requests.Response, dict]:
    """
    Runs property prediction using inference-ready (previously trained) ML models. And RDKIT funcs
    (They need to be calculated using this function, passing feature_column (if the user asks!)is:
    'Validity', 'Brenk', 'QED', 'Synthetic Accessibility', 'LogP', 'Polar Surface Area',
    'H-bond Donors', 'H-bond Acceptors', 'Rotatable Bonds', 'Aromatic Rings',
    'Glaxo', 'SureChEMBL', 'PAINS'.
    They are calculated automatically (by simply func) if they fall into arg: 'feature_column'.

    Args:
        smiles_list (List[str]): A list of molecules in SMILES format.
        case (str, optional): Name of model (model names can be obtained by calling 'get_state_from_server').
        timeout (int, optional): The timeout duration (in minutes).

    Returns:
        Tuple[requests.Response, dict]: A tuple containing the HTTP response object and the parsed JSON response.
    """
    url = "http://10.64.4.243:81/predict_ml"
    params = {"case": case, "smiles_list": smiles_list, "timeout": timeout}
    resp = requests.post(url, json.dumps(params))
    return resp, resp.json()


@tool
def train_ml_with_data(
    case="No_name",
    data_path="automl/data/data_4j1r.csv",  # path to client data folder
    feature_column=["Smiles"],
    target_column=[
        "Docking score"
    ],  # All propreties from dataframe you want to calculate in the end,
    regression_props=["Docking score"],
    classification_props=[],
    description="",
    timeout=5,
) -> bool:
    """
    Trains a predictive machine learning model using user-provided or prepared by a special agent dataset.

    This function reads a dataset from a specified file, processes it into a dictionary,
    and sends it to a remote server for training. The training process runs asynchronously
    using a separate process.

    FYI:
    RDKIT props (They need to be calculated using this function, passing feature_column (if the user asks!)is:
    'Validity', 'Brenk', 'QED', 'Synthetic Accessibility', 'LogP', 'Polar Surface Area',
    'H-bond Donors', 'H-bond Acceptors', 'Rotatable Bonds', 'Aromatic Rings',
    'Glaxo', 'SureChEMBL', 'PAINS'.
    They are calculated automatically (by simply func) if they fall into arg: 'feature_column'.

    Args:
        case (str, optional): Name of model.
        data_path (str, optional): Path to the CSV file containing the dataset.
        feature_column (list, optional): The name of the column containing the input features. Default is "Smiles".
        target_column (list, optional): All propreties from dataframe you want to calculate in the end. Default is "Polar Surface Area".
        regression_props (list, optional): Column names with data for regression tasks (That not include in RDKIT propss)
        classification_props (list, optional): Column name with data for classification tasks (That not include in RDKIT props)
        timeout (int, optional): The timeout duration (in minutes) for the request.
        description (str): Description of model/case

    Returns:
        bool (succces or not)"""
    start_time = time.time()
    try:
        df = pd.read_csv(
            data_path
        ).to_dict()  # Transfer df to dict for server data transfer
    except:
        return "Unable to open file. Invalid path to dataset."
    start_time = time.time()
    params = {
        "case": case,
        "data": df,
        "target_column": target_column,
        "feature_column": feature_column,
        "timeout": timeout,
        "description": description,
        "regression_props": regression_props,
        "classification_props": classification_props,
    }

    # Get state from server
    # state, is_exist = get_state_from_server(case=case, url="http://10.64.4.243:81/train_ml")

    # print(state['ml_state'])
    # print(state['calc_propreties'])
    # Get state from server
    # resp = requests.post(url,json.dumps(params))
    p = Process(
        target=requests.post,
        args=["http://10.64.4.243:81/train_ml", json.dumps(params)],
    )
    p.start()

    # time.sleep(10)
    # p.terminate()
    # resp = requests.post(url, data=json.dumps(params))
    print("--- %s seconds ---" % (time.time() - start_time))

    # succes = not(is_exist)
    # return succes


@tool
def get_state_from_server(url="http://10.64.4.243:81/", case=""):
    """Get dict with status about last ml-training from server. Run if need status for any of the models !!!

    Args:
        url (str): 'http://10.64.4.243:81/'

    Returns:
        [dict, bool]"""
    url_ = url.split("http://")[1]
    resp = requests.get("http://" + url_.split("/")[0] + "/check_state")
    is_exist = False

    if case in json.loads(resp.content)["ml_state"]:
        print("Case already trained!")
        is_exist = True

    state = json.loads(resp.content)["ml_state"]
    return state, is_exist


ml_dl_tools_rendered = render_text_description(
    [get_state_from_server, train_ml_with_data, predict_prop_by_smiles]
)

if __name__ == "__main__":
    st = train_ml_with_data(
        "Docking555",
        "/Users/alina/Desktop/ИТМО/ChemCoScientist/ChemCoScientist/dataset_handler/chembl/docking.csv",
    )
    st = get_state_from_server()
    print(st)
# res = get_state_from_server()
# print(res)

# res = predict_prop_by_smiles(['Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1'], "test_api")
# print(res)
