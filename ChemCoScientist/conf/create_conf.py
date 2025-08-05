import os

from protollm.connectors import create_llm_connector

from ChemCoScientist.agents.agents import (
    chemist_node,
    dataset_builder_agent,
    ml_dl_agent,
    nanoparticle_node,
    paper_analysis_agent,
)
from CoScientist.scientific_agents.agents import coder_agent
from ChemCoScientist.tools import chem_tools_rendered, nano_tools_rendered, tools_rendered, \
    paper_analysis_tools_rendered

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

# paper_analysis_node_description = (
#     "'paper_analysis_node' - answers questions by retrieving and analyzing information "
#     "from a database of chemical scientific papers. Using this agent takes precedence over web search."
# )

additional_agents_description = (
    automl_agent_description
    + dataset_builder_agent_description
    + coder_agent_description
    # + paper_analysis_node_description
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
            "paper_analysis_agent",
        ],
        # nodes for scenario agents
        "scenario_agent_funcs": {
            "chemist_node": chemist_node,
            "nanoparticle_node": nanoparticle_node,
            "ml_dl_agent": ml_dl_agent,
            "dataset_builder_agent": dataset_builder_agent,
            "coder_agent": coder_agent,
            "paper_analysis_agent": paper_analysis_agent,
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
            "paper_analysis_agent": [paper_analysis_tools_rendered],
            "web_search": [
                "You can use web search to find information on the internet. "
            ],
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
                "model_name": os.environ["SCENARIO_LLM_MODEL"],
                "url": os.environ["SCENARIO_LLM_URL"],
                "api_key": os.environ["OPENAI_API_KEY"],
                "ds_dir": os.environ["DS_STORAGE_PATH"],
            },
            "coder_agent": {
                "model_name": os.environ["SCENARIO_LLM_MODEL"],
                "url": os.environ["SCENARIO_LLM_URL"],
                "api_key": os.environ["OPENAI_API_KEY"],
                "ds_dir": os.environ["ANOTHER_STORAGE_PATH"],
            },
            "ml_dl_agent": {
                "model_name": os.environ["SCENARIO_LLM_MODEL"],
                "url": os.environ["SCENARIO_LLM_URL"],
                "api_key": os.environ["OPENAI_API_KEY"],
                "ds_dir": os.environ["DS_STORAGE_PATH"],
            },
        },
        # These prompts will be added as hints in ProtoLLM
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
                To find an answer, use the paper search first! NOT the web search!\
                If you are asked to generate a molecule, just schedule the generation using bl_dl_agent. \
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
                "additional_hints": "When planning or correct plan to process any data always indicate the path to file, look for the path in the 'past_steps'.",
            },
        },
    },
}
