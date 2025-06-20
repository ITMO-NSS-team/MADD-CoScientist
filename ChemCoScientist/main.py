import os
from dotenv import load_dotenv

load_dotenv('../config.env')

# os.environ["OPENAI_API_KEY"] = "KEY"
os.environ["PATH_TO_DATA"] = "tools/models/datasets/image_dataset_multi_filtered"
os.environ["PATH_TO_CVAE_CHECKPOINT"] = "tools/models/checkpoints/cvae/model.pt"
os.environ["PATH_TO_RESULTS"] = "tools/generation_results"


from protollm.agents.builder import GraphBuilder
from protollm.connectors import create_llm_connector

from ChemCoScientist.agents.agents import (
    chemist_node,
    ml_dl_agent,
    nanoparticle_node,
    dataset_builder_agent,
    paper_analysis_node,
)

from tools import (
    chem_tools_rendered,
    nano_tools_rendered,
    tools_rendered,
    dataset_handler_rendered,
)
from tools.ml_tools import ml_dl_tools_rendered
from CoScientist.scientific_agents.agents import coder_agent


model = create_llm_connector(
    "https://api.vsegpt.ru/v1;meta-llama/llama-3.1-70b-instruct"
)

visual_model = create_llm_connector(
    "https://api.vsegpt.ru/v1;vis-meta-llama/llama-3.2-90b-vision-instruct"
)
# description for agent WITHOUT langchain-tools
agent_rendered = "'dataset_builder_agent' - collects data from two databases - ChemBL and BindingDB. \
    To collect data, it needs either the protein name or a specific id from a specific database. \
        It can collect data from one specific database or from both. All data is saved locally. \
            It can also write simple processing code if asked. \
                'coder_agent' - can write any simple python scientific code. \
                    Can use rdkit and other chemical libraries. Can perform calculations. \
                'paper_analysis_node' - answers questions by retrieving and analyzing information from \
                    a database of chemical scientific papers."

conf = {
    # maximum number of recursions
    "recursion_limit": 50,
    "configurable": {
        "user_id": "1",
        "visual_model": visual_model,
        "img_path": "image.png",
        "llm": model,
        "max_retries": 1,
        # list of scenario agents
        "scenario_agents": [
            # "chemist_node",
            # "nanoparticle_node",
            # "ml_dl_agent",
            # "dataset_builder_agent",
            # "coder_agent",
            "paper_analysis_node",
        ],
        # nodes for scenario agents
        "scenario_agent_funcs": {
            # "chemist_node": chemist_node,
            # "nanoparticle_node": nanoparticle_node,
            # "ml_dl_agent": ml_dl_agent,
            # "dataset_builder_agent": dataset_builder_agent,
            # "coder_agent": coder_agent,
            "paper_analysis_node": paper_analysis_node,
        },
        # descripton for agents tools (if exist!!!), optional
        "tools_for_agents": {
            # here can be description of langchain web tools (not TavilySearch)
            # "web_serach": [web_tools_rendered],
            # "chemist_node": [chem_tools_rendered],
            # "nanoparticle_node": [nano_tools_rendered],
            # "ml_dl_agent": [ml_dl_tools_rendered],
            # "dataset_builder_agent": [dataset_handler_rendered],
        },
        # here can be langchain web tools (not TavilySearch)
        # "web_tools": web_tools,
        # full descripton for agents tools
        "tools_descp": tools_rendered,
        # description of agents (if they don't have tools) in free format
        "agents_descp": agent_rendered,
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
# inputs = {"input": "Обучи модель на данных из ChemBl предсказывать значение IC50. Модель сохрани с названием 'chembl_ic50'."}
# inputs = {"input": "Найди информацию о последних открытиях в области лечения Рака."}
# inputs = {"input": "Получи данные Ki по Q9BPZ7 из BindingDB."}
inputs = {"input": "Получи данные по KRAS G12C из доступных химических баз данных."}
# inputs = {
#     "input": "Посчитай sin(5) + 5837 / 544 + 55 * 453 + 77^4 с помощью агента-кодера"
# }
inputs = {"input": "How does the synthesis of Glionitrin A/B happen?"}


if __name__ == "__main__":
    graph = GraphBuilder(conf)
    res_1 = graph.run(inputs, debug=True, user_id="1")
