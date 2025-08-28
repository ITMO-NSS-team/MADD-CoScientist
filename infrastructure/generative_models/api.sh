#!/bin/bash
cd /projects/CoScientist/infrastructure/generative_models
source /root/miniconda3/bin/activate Mol_gen_env
nohup python main_api.py > api.txt
