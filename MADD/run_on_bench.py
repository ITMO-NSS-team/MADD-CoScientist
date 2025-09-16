from dotenv import load_dotenv
import pandas as pd
import os
load_dotenv("config.env")

from MADD.mas.create_graph import create_by_default_setup

if __name__ == "__main__":
    os.environ['DS_FROM_BINDINGDB'] = str(False)
    os.environ['DS_FROM_CHEMBL'] = str(False)
    os.environ['DS_FROM_USER'] = str(False)
    
    
    ds = pd.read_excel('benchmark/experiment1.xlsx').iterrows()
    
    graph = create_by_default_setup()
    for idx, sample in ds:
        print(f'Sample: {idx}.')
        print(f'Case: {sample["case"]}')
        for step in graph.stream(
            {
                "input": sample['content']
            },
            user_id="1",
        ):
            print(step)
