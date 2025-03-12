from logging import debug
from fastapi import FastAPI,Body
import os
import sys
sys.path.append(os.getcwd())
import uvicorn
import json
import_path = os.path.dirname(os.path.abspath(__file__))
import socket
import yaml
from api_utils import *
#sys.path.append('automl')

with open("automl/config.yaml", "r") as file:
    config = yaml.safe_load(file)

is_public_API = config['is_public_API']


if __name__=='__main__':
    def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('10.254.254.254', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP
    
    print("Getting IP")
    ip = str(get_ip())
    print(f"Current IP: {ip}")
    print("Starting...")

    app = FastAPI(debug=True)

    # API operations
    @app.get("/")
    def health_check():
        return {'health_check': 'OK'}

    @app.post("/train_ml")
    def train_ml_api(data:MLData=Body()):
        train_ml(data)

    @app.post("/predict_ml")
    def predict_ml_api(data:MLData=Body()):
        print(data)
        return json.dumps(inference_ml(data).tolist())
    

    if is_public_API:
        uvicorn_ip = ip
    else:
        uvicorn_ip = '127.0.0.1' 
    uvicorn.run(app,host=uvicorn_ip,port=81,log_level='info')
