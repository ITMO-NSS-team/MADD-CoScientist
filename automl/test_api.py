import requests
import json
import socket
import time 
import multiprocessing as mp 
from multiprocessing import Process
import os

def train_ml(
    case = "Brain_cancer",

    data_path = "automl/data/data_4j1r.csv",
    feature_column = 'canonical_smiles',
    target_column = 'docking_score',
    problem = 'regression',
    timeout = 10, #30 min
    url: str = "http://10.64.4.243:81/train_ml",
    **kwargs,
):
    start_time = time.time()
    
    params = {
        'case': case,
        'data_path': data_path,
        'target_column': target_column,
        'feature_column': feature_column,
        'problem': problem,
        'timeout': timeout,
        **kwargs,
    }
    p = Process(target=requests.post,args=(url, json.dumps(params)))
    p.start()
    time.sleep(1)
    p.terminate()
    #resp = requests.post(url, data=json.dumps(params))
    print("--- %s seconds ---" % (time.time() - start_time))
    #return resp, json.loads(resp.json())



if __name__=='__main__':
    train_ml()
    # p = Process(target=train_ml,)
    # p.start()
    print('Process created')
    