import logging
import os
from typing import Annotated
import operator

from langgraph.types import Command
from smolagents import CodeAgent, LiteLLMModel, OpenAIServerModel

from MADD.mas.prompts import (
    additional_ds_builder_prompt,
    automl_prompt,
    ds_builder_prompt,
)
from MADD.mas.utils import get_all_files
from MADD.mas.tools.data_gathering import fetch_BindingDB_data, fetch_chembl_data
from MADD.mas.tools.automl_tools import automl_tools


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dataset_builder_agent(state: dict, config: dict) -> Command:
    print("--------------------------------")
    print("Dataset builder agent called")
    print(state["task"])
    print("--------------------------------")
    task = state["task"]

    config_cur_agent = config["configurable"]["additional_agents_info"]["dataset_builder_agent"]
    print(config_cur_agent)

    model = (
        LiteLLMModel(config_cur_agent["model_name"], api_base=config_cur_agent["url"], api_key=config_cur_agent["api_key"])
        if "groq.com" in config_cur_agent["url"]
        else OpenAIServerModel(api_base=config_cur_agent["url"], model_id=config_cur_agent["model_name"], api_key=config_cur_agent["api_key"])
    )

    agent = CodeAgent(
        tools=[fetch_BindingDB_data, fetch_chembl_data],
        model=model,
        additional_authorized_imports=["*"],
    )

    response = agent.run(
        ds_builder_prompt + config_cur_agent["ds_dir"] + "\nSo, user ask: \n" + task + additional_ds_builder_prompt
    )

    files = get_all_files(os.environ["DS_STORAGE_PATH"])

    return Command(update={
        "past_steps": Annotated[set, operator.or_](set([(task, str(response))])),
        "nodes_calls": Annotated[set, operator.or_](set([
            ("dataset_builder_agent", (("text", str(response)),))
        ])),
        "metadata": Annotated[dict, operator.or_]({
            "dataset_builder_agent": files
        }),
    })


def ml_dl_agent(state: dict, config: dict) -> Command:
    print("--------------------------------")
    print("ml_dl agent called")
    print(state["task"])
    print("--------------------------------")

    task = state["task"]
    config_cur_agent = config["configurable"]["additional_agents_info"]["ml_dl_agent"]

    model = (
        LiteLLMModel(config_cur_agent["model_name"], api_base=config_cur_agent["url"], api_key=config_cur_agent["api_key"])
        if "groq.com" in config_cur_agent["url"]
        else OpenAIServerModel(api_base=config_cur_agent["url"], model_id=config_cur_agent["model_name"], api_key=config_cur_agent["api_key"])
    )

    agent = CodeAgent(
        tools=automl_tools,
        model=model,
        additional_authorized_imports=["*"],
    )
    response = agent.run(automl_prompt + task)

    return Command(update={
        "past_steps": Annotated[set, operator.or_](set([(task, str(response))])),
        "nodes_calls": Annotated[set, operator.or_](set([
            ("ml_dl_agent", (("text", str(response)),))
        ])),
    })
