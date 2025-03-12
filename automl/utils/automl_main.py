import os
import sys
from typing import List

from sklearn import metrics
sys.path.append(os.getcwd())
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from automl.utils.read_state import *
from fedot.api.main import Fedot,Pipeline
import logging
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup

def input_data_preparing(case:str,
                         task:str = 'regression',
                         split:bool = True):
    state = TrainState()
    df = pd.read_csv(state(case,'ml')['data']['data_path']).iloc[:1000,:]

    df_x = df[state(case,'ml')['data']['feature_column']].apply(lambda x: Chem.MolFromSmiles(x))
    df_x = df_x.dropna().apply(
        lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048)
    )
    X = np.array(df_x.tolist())
    y = df[state(case,'ml')['data']['target_column']]
    X = pd.DataFrame(data=X)
    y = pd.DataFrame(data=y)
    data = InputData.from_dataframe(X,
                                y,
                                task=task)
    data.supplementary_data.is_auto_preprocessed = True
    if split:
        train, test = train_test_data_setup(data)

        return  train, test
    else:
        return data
    
def input_data_preparing_from_list(case:str,
                                   data:List[str],
                                    split:bool = True,):
    state = TrainState()
    data = {"Smiles":data}
    df = pd.DataFrame(data=data)
    df_x = df['Smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df_x = df_x.dropna().apply(
        lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048)
    )
    X = np.array(df_x.tolist())
    #y = df[state(case,'ml')['data']['target_column']]
    X = pd.DataFrame(data=X)
    y = pd.DataFrame(data={})
    data = InputData.from_dataframe(X,y,
                                task=state(case,'ml')['data']['problem'])
    data.supplementary_data.is_auto_preprocessed = True
    if split:
        train, test = train_test_data_setup(data)

        return  train, test
    else:
        return data

def run_train_automl(case:str,
                     timeout:int = 5,
                     path_to_save = r'generative_models_data/generative_models/transformer_auto'):
    state = TrainState()

    train, test = input_data_preparing(case=case, task = state(case,'ml')['data']['problem'])
    available_secondary_operations = ['catboostreg','rfr', 'xgboostreg']
    if not os.path.isdir(path_to_save+f'_{case}'):
        os.mkdir(path_to_save+f'_{case}')
    if not os.path.isdir(path_to_save+f'_{case}'):
        os.mkdir(path_to_save+f'_{case}')
    state.ml_model_upd_status(case=case,model_weight_path=path_to_save+f'_{case}')
    model = Fedot(
        problem=state(case,'ml')['data']['problem'],
        preset='fast',  # Options: 'fast', 'stable', 'best_quality', etc.
        timeout=timeout,  # Minutes for optimization
        with_tuning=True,  # Allow tuning mode
        n_jobs=-1,  # CPU cores to use (-1 = all)
        cv_folds=5,  # Cross-validation folds
       available_operations = available_secondary_operations,
       #metric = ['mae']
       )
    model.fit(features=train.features, target=train.target)
    model.current_pipeline.save(path=path_to_save+f'_{case}', create_subdir=False, is_datetime_in_path=False)
    

    model.predict(features=test.features)
    state.ml_model_upd_status(case=case,metric=model.get_metrics(test.target))
    print(model.get_metrics(test.target))

#Test function
def run_predict_automl_from_data(case:str,
                     timeout:int = 5,
                     path_to_save = r'generative_models_data/generative_models/transformer_auto'):
    state = TrainState()
    data = input_data_preparing(case=case, task = state(case,'ml')['data']['problem'], split=False)
    pipeline = Pipeline().load(state(case,'ml')['weights_path'])
    #model.current_pipeline.save(path=path_to_save+f'_{case}', create_subdir=False, is_datetime_in_path=False)
    resutls = pipeline.predict(input_data=data)
    print(resutls)

def run_predict_automl_from_list(case:str,
                     data:List[str]):
    state = TrainState()
    data = input_data_preparing_from_list(case=case, split=False,data=data)
    pipeline = Pipeline().load(state(case,'ml')['weights_path'])
    resutls = pipeline.predict(input_data=data)
    return resutls.predict

if __name__=='__main__':

#####
#Example for train
    # state = TrainState()
    # task = "Brain_cancer"
    # data_path = "automl\data\data_4j1r.csv"
    # feature_column = 'canonical_smiles'
    # target_column = 'docking_score'
    # problem = 'regression'

    # state.add_new_case(case_name=task,rewrite=True)
    # state.ml_model_upd_data(case=task,
    #                         data_path=data_path,
    #                         feature_column=feature_column,
    #                         target_column=target_column,
    #                         problem=problem)
    
    # run_train_automl(case=task,path_to_save='automl/trained_data')
#######


######
#Example for inference
    task = "Brain_cancer"
    state = TrainState()
    smiles_list = pd.read_csv(state(task,'ml')['data']['data_path']).iloc[:1000,:]['canonical_smiles'].to_list()
    run_predict_automl_from_list(case=task,data=smiles_list)
