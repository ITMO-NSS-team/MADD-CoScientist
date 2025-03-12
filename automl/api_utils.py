from typing import List 
from fastapi import Body
from pydantic import BaseModel
from utils.read_state import TrainState
from automl.utils.automl_main import run_train_automl,run_predict_automl_from_list

class MLData(BaseModel):
        case:str = None
        data_path:str = None
        target_column:str = None
        problem:str = None
        smiles_list: list = None
        timeout:int = 30 #30 min
        feature_column:str = 'Smiles'
        path_to_save:str = 'automl/trained_data'

def train_ml(data:MLData=Body()):
        state = TrainState()
        state.add_new_case(case_name=data.case,rewrite=True)
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