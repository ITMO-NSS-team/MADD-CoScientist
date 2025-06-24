import os

from definitions import CONFIG_PATH
from dotenv import load_dotenv
load_dotenv(CONFIG_PATH)

from protollm.agents.builder import GraphBuilder
from protollm.connectors import create_llm_connector

from ChemCoScientist.agents.agents import (
    chemist_node,
    dataset_builder_agent,
    ml_dl_agent,
    nanoparticle_node,
    paper_analysis_node,
)
from CoScientist.scientific_agents.agents import coder_agent
from tools import chem_tools_rendered, nano_tools_rendered, tools_rendered

model = create_llm_connector(
    "https://api.vsegpt.ru/v1;meta-llama/llama-3.1-70b-instruct"
)

visual_model = create_llm_connector(
    "https://api.vsegpt.ru/v1;vis-meta-llama/llama-3.2-90b-vision-instruct"
)
# description for agent WITHOUT langchain-tools
automl_agent_description = """
'ml_dl_agent' - an agent that can run training of a generative model to generate SMILES, training of predictive models 
to predict properties. It also already stores ready-made models for inference. You can also ask him to prepare an 
existing dataset (you need to be specific in your request).
It can generate medicinal molecules. You must use this agent for molecules generation!!!\n

"""
dataset_builder_agent_description = "'dataset_builder_agent' - collects data from two databases - ChemBL and BindingDB. \
    To collect data, it needs either the protein name or a specific id from a specific database. \
        It can collect data from one specific database or from both. All data is saved locally. \
        It also processes data: removes junk values, empty cells, and can filter if necessary.\n"

coder_agent_description = "'coder_agent' - can write any simple python scientific code. Can use rdkit and other " \
                          "chemical libraries. Can perform calculations.\n "

paper_analysis_node_description = "'paper_analysis_node' - answers questions by retrieving and analyzing information " \
                                  "from a database of chemical scientific papers."

additional_agents_description = (
    automl_agent_description
    + dataset_builder_agent_description
    + coder_agent_description
    + paper_analysis_node_description
)

conf = {
    # maximum number of recursions
    "recursion_limit": 25,
    "configurable": {
        "user_id": "1",
        "visual_model": visual_model,
        "img_path": "image.png",
        "llm": model,
        "max_retries": 1,
        # list of scenario agents
        "scenario_agents": [
            "chemist_node",
            "nanoparticle_node",
            "ml_dl_agent",
            "dataset_builder_agent",
            "coder_agent",
            "paper_analysis_node",
        ],
        # nodes for scenario agents
        "scenario_agent_funcs": {
            "chemist_node": chemist_node,
            "nanoparticle_node": nanoparticle_node,
            "ml_dl_agent": ml_dl_agent,
            "dataset_builder_agent": dataset_builder_agent,
            "coder_agent": coder_agent,
            "paper_analysis_node": paper_analysis_node,
        },
        # descripton for agents tools - if using langchain @tool
        # or description of agent capabilities in free format
        "tools_for_agents": {
            # here can be description of langchain web tools (not TavilySearch)
            # "web_serach": [web_tools_rendered],
            "chemist_node": [chem_tools_rendered],
            "nanoparticle_node": [nano_tools_rendered],
            "dataset_builder_agent": [dataset_builder_agent_description],
            "coder_agent": [coder_agent_description],
            "ml_dl_agent": [automl_agent_description],
            "paper_analysis_node": [paper_analysis_node_description],
        },
        # here can be langchain web tools (not TavilySearch)
        # "web_tools": web_tools,
        # full descripton for agents tools
        "tools_descp": tools_rendered + additional_agents_description,
        # set True if you want to use web search like black-box
        "web_search": True,
        # add a key with the agent node name if you need to pass something to it
        "additional_agents_info": {
            "dataset_builder_agent": {
                "model_name": "deepseek/deepseek-chat-0324-alt-structured",
                "url": "https://api.vsegpt.ru/v1",
                "api_key": os.environ["OPENAI_API_KEY"],
                #  Change on your dir if another!
                "ds_dir": "./data_dir_for_coder",
            },
            "coder_agent": {
                "model_name": "deepseek/deepseek-chat-0324-alt-structured",
                "url": "https://api.vsegpt.ru/v1",
                "api_key": os.environ["OPENAI_API_KEY"],
                #  Change on your dir if another!
                "ds_dir": "./data_dir_for_coder",
            },
            "ml_dl_agent": {
                "model_name": "deepseek/deepseek-chat-0324-alt-structured",
                "url": "https://api.vsegpt.ru/v1",
                "api_key": os.environ["OPENAI_API_KEY"],
                #  Change on your dir if another!
                "ds_dir": "./data_dir_for_coder",
            },
        },
        # These prompts will be added as hints in ProtoLLM
        "prompts": {
            "planner": "Before you start training models, plan to check your data for garbage using a dataset_builder_agent",
            "chat": """You are a chemical agent system. You can do the following:
                    - train generative models (generate SMILES molecules), train predictive models (predict properties)
                    - prepare a dataset for training
                    - download data from chemical databases: ChemBL, BindingDB
                    - perform calculations with chemical python libraries
                    - solve problems of nanomaterial synthesis
                    - analyze chemical articles
                    """
        },
    },
}


# UNCOMMENT the one you need

# inputs = {"input": "Посчитай qed, tpsa, logp, hbd, hba свойства ацетона"}
# inputs = {"input": "Поищи с помощью поиска свежие статьи на тему онлайн-синтеза наноматериалов"}
# inputs = {"input": "Generate an image for nanoparticles most optimal for catalysis."}
# inputs = {"input": "Что ты можешь?"}
# inputs = {"input": "Узнай есть ли обученные модели, напиши какие именно доступны."}
# inputs = {"input": "Запусти предсказание с помощью мл-модели на значение IC50 для молекулы Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1."}
# inputs = {"input": "Запусти обучение на данных из /Users/alina/Desktop/ИТМО/ChemCoScientist/ChemCoScientist/dataset_handler/chembl/docking.csv. В качестве таргета возьми Docking score. Обязательно назови кейс DOCKING_SCORE"}
# inputs = {"input": "Предскажи Docking score для Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1 с помощью мл-модели."}
# inputs = {"input": "Модель с названием DOCKING_SCORE еще обучается?"}
# inputs = {"input": "Найди информацию о последних открытиях в области лечения Рака."}
# inputs = {"input": "Получи данные Ki по Q9BPZ7 из BindingDB."}
# inputs = {"input": "Получи данные по KRAS G12C из доступных химических баз данных."}
# inputs = {"input": "Сгенерируй мне какие-нибудь молекулы."}
# inputs = {"input": "Сгенерируй молекулы для лечения Alzheimer."}

# inputs = {
#     "input": "Посчитай sin(5) + 5837 / 544 + 55 * 453 + 77^4 с помощью агента-кодера"
# }

# inputs = {"input": "Запусти обучение генеративной модели на данных '/Users/alina/Desktop/ИТМО/ChemCoScientist/data_dir_for_coder/chembl_ic50_data.xlsx', назови кейс IC50_chembl."}
# inputs = {"input": "Какой статус обучения у кейса Docking_hight?"}
# inputs = {"input": "Запусти предсказание с помощью мл-модели на значение IC50 для молекулы Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1."}
inputs = {"input": "How does the synthesis of Glionitrin A/B happen based on research?"}


if __name__ == "__main__":
    graph = GraphBuilder(conf)
    # while True:
    #     task = input()
    #     res_1 = graph.run({"input": task}, debug=True, user_id="1")
    res_1 = graph.run(inputs, debug=True, user_id="1")
