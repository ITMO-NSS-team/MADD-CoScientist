from typing import List 
from fastapi import Body
import os
import sys
from pydantic import BaseModel
from utils.read_state import TrainState
from utils.automl import run_train_automl

class MLData(BaseModel):
        case:str 
        data_path:str 
        target_column:str
        problem:str
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
        pass