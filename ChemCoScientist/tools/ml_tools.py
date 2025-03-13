from langchain_core.tools import tool
from typing import List,Tuple
import requests
import json
import socket
import time 
import multiprocessing as mp 
from multiprocessing import Process
import os
import pandas as pd

# TODO: make yaml conf
conf = {'url': "http://10.64.4.243:81"}

@tool
def predict_prop_by_smiles(smiles_list : List[str],
                   case : str = "No_name",
                   timeout : int = 10) -> Tuple[requests.Response, dict]:
    """
    Runs property prediction using inference-ready (previously trained) ML models. 
    
    Args:
        smiles_list (List[str]): A list of molecules in SMILES format.
        case (str, optional): Name of model.
        timeout (int, optional): The timeout duration (in minutes).
        
    Returns:
        Tuple[requests.Response, dict]: A tuple containing the HTTP response object and the parsed JSON response.
    """
    params = {
        'case': case,
        'smiles_list' : smiles_list,
        'timeout': timeout
    }
    resp = requests.post(conf['url'] + '/predict_ml', json.dumps(params))
    return resp, json.loads(resp.json())

@tool
def train_ml_with_data(
    case = "No_name",
    data_path = "automl/data/data_4j1r.csv", #path to client data folder
    feature_column = 'Smiles',
    target_column = 'Polar Surface Area',
    problem = 'regression',
    description = '',
    timeout = 10
) -> bool:
    """
    Trains a predictive machine learning model using user-provided or prepared by a special agent dataset.

    This function reads a dataset from a specified file, processes it into a dictionary, 
    and sends it to a remote server for training. The training process runs asynchronously 
    using a separate process.

    Args:
        case (str, optional): Name of model.
        data_path (str, optional): Path to the CSV file containing the dataset.
        feature_column (str, optional): The name of the column containing the input features. Default is "Smiles".
        target_column (str, optional): The name of the column containing the target variable. Default is "Polar Surface Area".
        problem (str, optional): The type of problem ("regression" or "classification"). Default is "regression".
        description (str, optional): A description of the training case. Default is "Case for Brain cancer".
        timeout (int, optional): The timeout duration (in minutes) for the request.

    Returns:
        bool (succces or not)"""
    start_time = time.time()
    try:
        df = pd.read_csv(data_path).to_dict() # Transfer df to dict for server data transfer
    except:
        return 'Unable to open file. Invalid path to dataset.'
    params = {
        'case': case,
        "data" : df,
        'target_column': target_column,
        'feature_column': feature_column,
        'problem': problem,
        'timeout': timeout,
        'description' : description
    }

    #Get state from server
    state, is_exist = get_state_from_server(conf['url'] + '/train_ml') 
    #Get state from server

    p = Process(target=requests.post, kwargs={"url": conf['url'] + '/train_ml', "data": json.dumps(params)})
    p.start()
    time.sleep(5)
    p.terminate()

    print("--- %s seconds ---" % (time.time() - start_time))
    
    succes = not(is_exist)
    return succes
    
    
    
@tool
def get_state_from_server(url = "http://10.64.4.243:81/", case = None):
    """Get dict with status about last ml-training from server. Run if need status for any of the models !!!
    
    Args:
        url (str): 'http://10.64.4.243:81/' 
        
    Returns:
        [dict, bool]"""
    url_ = url.split("http://")[1]
    resp = requests.get("http://"+url_.split('/')[0]+"/")
    is_exist = False

    if case in json.loads(resp.content)['ml_state']:
        print("Case already trained!")
        is_exist = True
    
    state = json.loads(resp.content)['ml_state']
    return state, is_exist
