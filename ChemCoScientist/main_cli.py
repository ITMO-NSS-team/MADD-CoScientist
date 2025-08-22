import os

from definitions import CONFIG_PATH
from dotenv import load_dotenv

load_dotenv(CONFIG_PATH)

from protollm.agents.builder import GraphBuilder
from protollm.connectors import create_llm_connector
from protollm.agents.universal_agents import web_search_node

from ChemCoScientist.agents.agents import (
    chemist_node,
    dataset_builder_agent,
    ml_dl_agent,
    nanoparticle_node,
    paper_analysis_agent,
)
from CoScientist.scientific_agents.agents import coder_agent
from ChemCoScientist.tools import chem_tools_rendered, nano_tools_rendered, tools_rendered, paper_analysis_tools_rendered
from definitions import ROOT_DIR

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

coder_agent_description = (
    "'coder_agent' - can write any simple python scientific code. Can use rdkit and other "
    "chemical libraries. Can perform calculations.\n "
)

paper_analysis_node_description = (
    "'paper_analysis_node' - answers questions by retrieving and analyzing information "
    "from a database of chemical scientific papers. Using this agent takes precedence over web search."
)
web_search_description = "You can use web search to find information on the internet. "

additional_agents_description = (
    automl_agent_description
    + dataset_builder_agent_description
    + coder_agent_description
    + paper_analysis_node_description
    + web_search_description
)

conf = {
    # maximum number of recursions
    "recursion_limit": 25,
    "configurable": {
        "user_id": "1",
        "visual_model": create_llm_connector(os.environ["VISION_LLM_URL"]),
        "img_path": "image.png",
        "llm": create_llm_connector(
            f"{os.environ['MAIN_LLM_URL']};{os.environ['MAIN_LLM_MODEL']}"
        ),
        "max_retries": 3,
        # list of scenario agents
        "scenario_agents": [
            "chemist_node",
            "nanoparticle_node",
            "ml_dl_agent",
            "dataset_builder_agent",
            "coder_agent",
            "paper_analysis_node",
            "web_search"
        ],
        # nodes for scenario agents
        "scenario_agent_funcs": {
            "chemist_node": chemist_node,
            "nanoparticle_node": nanoparticle_node,
            "ml_dl_agent": ml_dl_agent,
            "dataset_builder_agent": dataset_builder_agent,
            "coder_agent": coder_agent,
            "paper_analysis_node": paper_analysis_agent,
            "web_search": web_search_node
        },
        # descripton for agents tools - if using langchain @tool
        # or description of agent capabilities in free format
        "tools_for_agents": {
            "chemist_node": [chem_tools_rendered],
            "nanoparticle_node": [nano_tools_rendered],
            "dataset_builder_agent": [dataset_builder_agent_description],
            "coder_agent": [coder_agent_description],
            "ml_dl_agent": [automl_agent_description],
            "paper_analysis_node": [paper_analysis_tools_rendered],
            "web_search": [web_search_description],
        },
        # full descripton for agents tools
        "tools_descp": tools_rendered + additional_agents_description,
        # add a key with the agent node name if you need to pass something to it
        "additional_agents_info": {
            "dataset_builder_agent": {
                "model_name": os.environ["SCENARIO_LLM_MODEL"],
                "url": os.environ["SCENARIO_LLM_URL"],
                "api_key": os.environ["OPENAI_API_KEY"],
                "ds_dir": os.path.join(ROOT_DIR, os.environ["DS_STORAGE_PATH"]),
            },
            "coder_agent": {
                "model_name": os.environ["SCENARIO_LLM_MODEL"],
                "url": os.environ["SCENARIO_LLM_URL"],
                "api_key": os.environ["OPENAI_API_KEY"],
                "ds_dir": os.path.join(ROOT_DIR, os.environ["ANOTHER_STORAGE_PATH"]),
            },
            "ml_dl_agent": {
                "model_name": os.environ["SCENARIO_LLM_MODEL"],
                "url": os.environ["SCENARIO_LLM_URL"],
                "api_key": os.environ["OPENAI_API_KEY"],
                "ds_dir": os.path.join(ROOT_DIR, os.environ["DS_STORAGE_PATH"]),
            },
        },
        # These prompts will be added in ProtoLLM
        "prompts": {
            "supervisor": {
                "problem_statement": None,
                "problem_statement_continue": None,
                "rules": None,
                "additional_rules": None,
                "examples": None,
                "enhancemen_significance": None,
            },
            "planner": {
                "problem_statement": None,
                "rules": None,
                "desc_restrictions": None,
                "examples": None,
                "additional_hints": "Before you start training models, plan to check your data for garbage using a dataset_builder_agent.\n \
                If the user provides his dataset - immediately start training using ml_dl_agent (never call dataset_builder_agent)!\
                        To find an answer, use the paper search first! NOT the web search!",
            },
            "chat": {
                "problem_statement": None,
                "additional_hints": """You are a chemical agent system. You can do the following:
                    - train generative models (generate SMILES molecules), train predictive models (predict properties)
                    - prepare a dataset for training
                    - download data from chemical databases: ChemBL, BindingDB
                    - perform calculations with chemical python libraries
                    - solve problems of nanomaterial synthesis
                    - analyze chemical articles
                    
                    If user ask something like "What can you do" - make answer yourself!
                    """,
            },
            "summary": {
                "problem_statement": None,
                "rules": None,
                "additional_hints": "Never write full paths! Only file names.",
            },
            "replanner": {
                "problem_statement": None,
                "rules": None,
                "examples": None,
                "additional_hints": "Optimize the plan, transfer already existing answers from previous executions! For example, weather values.\
                Don't forget tasks! Plan the Coder Agent to save files.\
                    Be more careful about which tasks can be performed in parallel and which ones can be performed sequentially.\
                        For example, you cannot fill a table and save it in parallel.",
            },
        },
    },
}



# UNCOMMENT the one you need

# inputs = {"input": "Посчитай qed, tpsa, logp, hbd, hba свойства ацетона"}
# inputs = {"input": "Поищи с помощью веб-поиска свежие статьи на тему новых наноматериалов"}
# inputs = {"input": "Generate an image for nanoparticles most optimal for catalysis."}
# inputs = {"input": "Что ты можешь?"}
# inputs = {"input": "Узнай есть ли обученные модели, напиши какие именно доступны."}
# inputs = {"input": "Запусти предсказание с помощью мл-модели на значение IC50 для молекулы Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1."}
# inputs = {"input": "Запусти обучение на данных из /Users/alina/Desktop/ИТМО/ChemCoScientist/ChemCoScientist/dataset_handler/chembl/docking.csv. В качестве таргета возьми Docking score. Обязательно назови кейс DOCKING_SCORE"}
# inputs = {"input": "Предскажи Docking score для Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1 с помощью мл-модели."}
# inputs = {"input": "Модель с названием DOCKING_SCORE еще обучается?"}
# inputs = {"input": "Найди информацию о последних открытиях в области лечения Рака."}
# inputs = {"input": "Получи данные по KRAS G12C из доступных химических баз данных."}
# inputs = {"input": "Сгенерируй мне какие-нибудь молекулы."}
# inputs = {"input": "Сгенерируй молекулы для лечения Alzheimer."}

# inputs = {
#     "input": "Посчитай sin(5) + 5837 / 544 + 55 * 453 + 77^4 с помощью агента-кодера"
# }

# inputs = {"input": "Запусти обучение генеративной модели на данных '/Users/alina/Desktop/ИТМО/ChemCoScientist/data_dir_for_coder/chembl_ic50_data.xlsx', назови кейс IC50_chembl."}
# inputs = {"input": "What can you do?"}
# inputs = {"input": "Запусти предсказание с помощью мл-модели на значение IC50 для молекулы Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1."}
inputs = {"input": "How does the synthesis of Glionitrin A/B happen based on research?"}
# inputs = {"input": "what papers have info on the Synthesis of Glionitrin A/B?"}
# inputs = {"input": "what is the name of figure 1?"}
# inputs = {"input": "How does the synthesis of Glionitrin A/B happen based on research?"}

# parallel examples
# inputs = {"input": "Получи данные по KRAS G12C из ChemBL для Ki. Получи данные по MEK1 из ChemBL по Ki"}
# inputs = {"input": "Получи данные по KRAS G12C из ChemBL для IC50. Получи данные по MEK1 из ChemBL по IC50. Поставь обучение на том датасете, где данных больше, назови кейс в зависимости от белка."}
# inputs = {"input": "Получи данные по MEK1 из BindigDB для IC50. Получи данные по KRAS из BindigDB по Ki."}

graph = GraphBuilder(conf)

if __name__ == "__main__":
    for step in graph.stream(inputs, user_id="1"):
        print(step)
