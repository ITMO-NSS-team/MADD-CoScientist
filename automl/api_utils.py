from typing import List 
from fastapi import Body
from pydantic import BaseModel
from utils.read_state import TrainState
from automl.utils.automl_main import run_train_automl,run_predict_automl_from_list
import pandas as pd
import os 

class MLData(BaseModel):
        data:dict = None
        case:str = None
        data_path:str = None
        target_column:str = None
        problem:str = None
        smiles_list: list = None
        timeout:int = 30 #30 min
        feature_column:str = 'Smiles'
        path_to_save:str = 'automl/trained_data'
        description:str = 'Unknown case.'

def train_ml_with_data(data:MLData=Body()):
        state = TrainState()
        state.add_new_case(case_name=data.case,
                           rewrite=True,
                           description=data.description)
        if data.data is not None:
                df = pd.DataFrame(data.data)
                data.data_path = f"automl/data/{data.case}"
                if not os.path.isdir(data.data_path):
                    os.mkdir(data.data_path)
                data.data_path = data.data_path + '/data.csv'
                df.to_csv(data.data_path)       
        state.ml_model_upd_data(case=data.case,
                                data_path=data.data_path,
                                feature_column=data.feature_column,
                                target_column=data.target_column,
                                problem=data.problem)
        run_train_automl(case=data.case,
                         path_to_save=data.path_to_save,
                         timeout=data.timeout)

def inference_ml(data:MLData=Body()):
        resutls = run_predict_automl_from_list(data.case,data=data.smiles_list)
        return resutls