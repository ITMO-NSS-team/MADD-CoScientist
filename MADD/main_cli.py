from dotenv import load_dotenv
import os
load_dotenv("config.env")

from MADD.mas.create_graph import create_by_default_setup

if __name__ == "__main__":
    os.environ['DS_FROM_BINDINGDB'] = str(False)
    os.environ['DS_FROM_CHEMBL'] = str(False)
    os.environ['DS_FROM_USER'] = str(False)
    
    graph = create_by_default_setup()
    for step in graph.stream(
        {
            "input": "Download data for KRAS with IC50."
        },
        user_id="1",
    ):
        print(step)
