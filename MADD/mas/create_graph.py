import os

from definitions import CONFIG_PATH
from dotenv import load_dotenv

load_dotenv(CONFIG_PATH)

from protollm.agents.builder import GraphBuilder
from protollm.connectors import create_llm_connector

from MADD.mas.scenarion_agents import (
    dataset_builder_agent, ml_dl_agent
)
from MADD.mas.prompts import automl_agent_description, dataset_builder_agent_description


def create_by_default_setup() -> GraphBuilder:
    functional_description = automl_agent_description + dataset_builder_agent_description
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
                "ml_dl_agent",
                "dataset_builder_agent",
            ],
            # nodes for scenario agents
            "scenario_agent_funcs": {
                "ml_dl_agent": ml_dl_agent,
                "dataset_builder_agent": dataset_builder_agent,
            },
            # descripton for agents tools - if using langchain @tool
            # or description of agent capabilities in free format
            "tools_for_agents": {
                "dataset_builder_agent": [dataset_builder_agent_description],
                "ml_dl_agent": [automl_agent_description],
            },
            # full descripton for agents tools
            "tools_descp": functional_description,
            # add a key with the agent node name if you need to pass something to it
            "additional_agents_info": {
                "dataset_builder_agent": {
                    "model_name": os.environ["SCENARIO_LLM_MODEL"],
                    "url": os.environ["SCENARIO_LLM_URL"],
                    "api_key": os.environ["OPENAI_API_KEY"],
                    "ds_dir": os.environ["DS_STORAGE_PATH"],
                },
                "ml_dl_agent": {
                    "model_name": os.environ["SCENARIO_LLM_MODEL"],
                    "url": os.environ["SCENARIO_LLM_URL"],
                    "api_key": os.environ["OPENAI_API_KEY"],
                    "ds_dir": os.environ["DS_STORAGE_PATH"],
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
                    "additional_hints": "If the user provides his dataset - \
                        immediately start training using ml_dl_agent (never call dataset_builder_agent)!",
                },
                "chat": {
                    "problem_statement": None,
                    "additional_hints": """You are a chemical agent system. You can do the following:
                        - train generative models (generate SMILES molecules), train predictive models (predict properties)
                        - prepare a dataset for training
                        - download data from chemical databases: ChemBL, BindingDB
                        - generate molecules that treat Alzheimer, Sclerosis, Lung Cancer, Dislipedimiya.
                        
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
    return GraphBuilder(conf)

