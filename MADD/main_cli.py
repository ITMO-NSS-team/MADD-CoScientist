from dotenv import load_dotenv
import os
load_dotenv("config.env")

from MADD.mas.create_graph import create_by_default_setup

if __name__ == "__main__":
    os.environ['DS_FROM_BINDINGDB'] = str(False)
    os.environ['DS_FROM_CHEMBL'] = str(False)
    os.environ['DS_FROM_USER'] = str(False)
    # UNCOMMENT, here path to dataset if you want to use it
    # os.environ['DS_FROM_USER'] = '/Users/alina/Desktop/ITMO/MADD-CoScientist/data_cyk_short.csv'
        
    graph = create_by_default_setup()
    while True:
        for step in graph.stream(
            {
                # "input": 'What models are available now?'
                # "input": 'Download data from ChemBL for GSK with IC50 values.'
                # "input": 'Start train model to predict PAINS, use my data.'
                # "input": "Generate 2 molecules by 'PAINS_predictor'."
                # "input": "Check status of training for generative model with case 'PAINS_predictor'"
                "input":  'Download data from ChemBL for GSK with IC50 values.'
            },
            user_id="1",
        ):
            print(step)
            
        