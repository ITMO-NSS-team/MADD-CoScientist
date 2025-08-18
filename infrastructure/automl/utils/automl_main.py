import os
import sys
import time
from typing import List

from annotated_types import T
sys.path.append(os.getcwd())
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from infrastructure.automl.utils.base_state import *
from fedot.api.main import Fedot,Pipeline
import logging
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
import itertools
from huggingface_hub import HfApi
from sklearn.metrics import (
    f1_score as f1,
    accuracy_score as accuracy,
    mean_absolute_error as mae,
    mean_squared_error as mse,
    r2_score as r2
)

def input_data_preparing(case:str,
                         problem:str = 'regression',
                         split:bool = True):
    state = TrainState()
    df = pd.read_csv(state(case,'ml')['data_path'])#.iloc[:1000,:]
    if type(state(case,'ml')['feature_column'])==str:
        df_x = df[state(case,'ml')['feature_column']].apply(lambda x: Chem.MolFromSmiles(x))
    elif type(state(case,'ml')['feature_column'])==list:
        df_x = df[state(case,'ml')['feature_column'][0]].apply(lambda x: Chem.MolFromSmiles(x))
    df_x = df_x.dropna().apply(
        lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048)
    )
    X = np.array(df_x.tolist())
    y = df[state(case,'ml')["Predictable properties"][problem]]
    X = pd.DataFrame(data=X)
    y = pd.DataFrame(data=y)
    data = InputData.from_dataframe(X,
                                y,task=problem)
    data.supplementary_data.is_auto_preprocessed = True
    if split:
        train, test = train_test_data_setup(data)

        return  train, test
    else:
        return data
    
def input_data_preparing_from_list(case:str,
                                   data:List[str],
                                    split:bool = True,
                                    problem:str='regression'):
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
                                task=problem)
    data.supplementary_data.is_auto_preprocessed = True
    if split:
        train, test = train_test_data_setup(data)

        return  train, test
    else:
        return data

def run_train_automl(case:str,
                     timeout:int = 60*20,
                     path_to_save = r'generative_models_data/generative_models/transformer_auto',
                     save_trained_data_to_sync_server:bool=False):
    state = TrainState()
    metrics = {'regression':None,'classification':None}
    state.ml_model_upd_status(case=case,status=1)
    for problem in state(case,'ml')["Predictable properties"]:
        train, test = input_data_preparing(case=case, problem = problem)

        #Uncomment to set available_operations with few models
        # if problem == 'regression':
        #     available_secondary_operations = ['catboostreg','rfr', 'xgboostreg']
        # elif problem == 'classification':
        #     available_secondary_operations = ['catboost','rf', 'xgboost']
        if not os.path.isdir(path_to_save+f'_{case}'+f'_{problem}'):
            os.mkdir(path_to_save+f'_{case}'+f'_{problem}')
        state.ml_model_upd_status(case=case,model_weight_path=path_to_save+f'_{case}'+f'_{problem}',problem=problem)
        model = Fedot(
            problem=problem,
            #preset='fast',  # Options: 'fast', 'stable', 'best_quality', etc.
            timeout=timeout,  # Minutes for optimization
            with_tuning=True,  # Allow tuning mode
            n_jobs=-1,  # CPU cores to use (-1 = all)
            cv_folds=5,  # Cross-validation folds
            pop_size = 10
        #available_operations = available_secondary_operations,

        )
        
        model.fit(features=train.features, target=train.target)
        model.current_pipeline.save(path=path_to_save+f'_{case}'+f'_{problem}', create_subdir=False, is_datetime_in_path=False)
        

        predict = model.predict(features=test.features)
        #metrics[problem] = model.get_metrics(test.target)
        metrics_dict = {}
        if problem == 'classification':
           
            metrics_dict[f"Accuracy score"]=accuracy(test.target, predict)
            metrics_dict[f"F1 score"]=f1(test.target, predict)
        elif problem == 'regression':
            metrics_dict[f"MAE score"]=mae(test.target, predict)
            metrics_dict[f"MSE score"]=mse(test.target, predict)
            metrics_dict[f"R2 score"]=r2(test.target, predict)
        else:
            ...

        metrics[problem] = metrics_dict

        if save_trained_data_to_sync_server:
            time.sleep(2)
            api = HfApi(token="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")
            last_folder = path_to_save.split(sep='/')[-1]
            api.upload_folder(repo_id='SoloWayG/Molecule_transformer',
                              folder_path=path_to_save+f'_{case}'+f'_{problem}',
                              path_in_repo=f'ML_models/{last_folder}_{case}_{problem}',
                              commit_message=f'Add model for {case} case with {problem} problem.',
                              #delete_patterns=True
            )
    state.ml_model_upd_status(case=case,metric=metrics,status=2)
    
        

#Test function
# def run_predict_automl_from_data(case:str,
#                      timeout:int = 5,
#                      path_to_save = r'generative_models_data/generative_models/transformer_auto'):
#     state = TrainState()
#     data = input_data_preparing(case=case, task = state(case,'ml')['problem'], split=False)
#     pipeline = Pipeline().load(state(case,'ml')['weights_path'])
#     #model.current_pipeline.save(path=path_to_save+f'_{case}', create_subdir=False, is_datetime_in_path=False)
#     resutls = pipeline.predict(input_data=data)
#     print(resutls)

def run_predict_automl_from_list(case:str,
                     data:List[str]):
    state = TrainState()
    properties = {}
    # state.ml_model_upd_data(case=case,
    #                         target_column=['docking_score','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50'],
    #                        )
    predicteble_props = [state(case,'ml')['Predictable properties'][i] for i in state(case,'ml')['Predictable properties']]
    predicteble_props = list(itertools.chain(*predicteble_props))
    calc_props = [i for i in state(case,'ml')['target_column'] if i not in predicteble_props]
    for p in calc_props:
        properties[p] = state()['Calculateble properties'][p](data)

    for problem in state(case,'ml')["Predictable properties"]:

        data_preapred = input_data_preparing_from_list(case=case, split=False,data=data,problem=problem)
        pipeline = Pipeline().load(state(case,'ml')['weights_path'][problem])
        resutls = pipeline.predict(input_data=data_preapred)
        for i,prop in enumerate(state(case,'ml')["Predictable properties"][problem]):
            if len(resutls.predict.shape)>1:
                properties[prop] = list(map(float,resutls.predict[:,i]))
            else:
                properties[prop] = list(map(float,resutls.predict))

    return properties



if __name__=='__main__':

#####
#Example for train
    state = TrainState()

    data_path = r"D:\Projects\CoScientist\automl\utils\merged_result2_train.csv"
    task = 'Alzheimer_regression_Minimum_Energy_Rodion'
    feature_column = ['canonical_smiles']
    target_column = ['Minimum Energy']
    regression_props = ['Minimum Energy']
    #classification_props = ['IC50']

    state.add_new_case(task,rewrite=True)
    state.ml_model_upd_data(case=task,
                            data_path = data_path,
                            feature_column=feature_column,
                            target_column=target_column,
                            predictable_properties={ "regression":regression_props})
    
    run_train_automl(case=task,path_to_save='automl/trained_data',timeout=60*10)


    # data_path = r"automl\data\base_cases\tau_kinase_inhib_reg.csv"
    # task = 'Alzheimer_regression'
    # feature_column = ['Smiles']
    # target_column = ['Standard Value']
    # regression_props = ['Standard Value']
    # #classification_props = ['IC50']

    # state.add_new_case(task,rewrite=True)
    # state.ml_model_upd_data(case=task,
    #                         data_path = data_path,
    #                         feature_column=feature_column,
    #                         target_column=target_column,
    #                         predictable_properties={ "regression":regression_props})
    
    # run_train_automl(case=task,path_to_save='automl/trained_data',timeout=2*60)

    #Log Standard Value
#######


######
#Example for inference
    # task = "Alzheimer_Minimum_Energy"
    # state = TrainState()
    # path_csv = r'D:\Projects\CoScientist\automl\data\data_4j1r.csv'
    # df = pd.read_csv(path_csv)
    # smiles_list = df['canonical_smiles'].to_list()
    # df['Log Standard Value']=run_predict_automl_from_list(case=task,data=smiles_list)['Log Standard Value']
    # print(df['Log Standard Value'])
    # df.to_csv(path_csv)
