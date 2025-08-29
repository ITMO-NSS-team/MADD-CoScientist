import logging
import operator
import os
from typing import Annotated

from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from MADD.mas.prompts.prompts import (
    automl_prompt,
    ds_builder_prompt,
)
from MADD.mas.tools.automl_tools import automl_tools
from MADD.mas.tools.data_gathering import fetch_BindingDB_data, fetch_chembl_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dataset_builder_agent(state: dict, config: dict) -> Command:
    print("--------------------------------")
    print("Dataset builder agent called")
    print(state["task"])
    print("--------------------------------")
    task = state["task"]

    agent = create_react_agent(
        config["configurable"]["llm"],
        [fetch_BindingDB_data, fetch_chembl_data],
        state_modifier=ds_builder_prompt,
        debug=True,
    )
    task_formatted = f"""\nYou are tasked with executing: {task}."""

    response = agent.invoke({"messages": [("user", task_formatted)]})
    
    ds_paths = [i for i in [os.environ.get('DS_FROM_CHEMBL', ''), os.environ.get('DS_FROM_BINDINGDB', '')] if i != '']


    return Command(
        update={
            "past_steps": Annotated[set, operator.or_](set([(task, response["messages"][-1].content)])),
            "nodes_calls": Annotated[set, operator.or_](
                set([("dataset_builder_agent", (("text", response["messages"][-1].content),))])
            ),
            "metadata": Annotated[dict, operator.or_]({"dataset_builder_agent": ds_paths}),
        }
    )


def ml_dl_agent(state: dict, config: dict) -> Command:
    print("--------------------------------")
    print("ml_dl agent called")
    print(state["task"])
    print("--------------------------------")

    task = state["task"]

    agent = create_react_agent(
        config["configurable"]["llm"],
        automl_tools,
        state_modifier=automl_prompt,
        debug=True,
    )
    task_formatted = f"""\nYou are tasked with executing: {task}."""

    response = agent.invoke({"messages": [("user", task_formatted)]})

    return Command(
        update={
            "past_steps": Annotated[set, operator.or_](
                set([(task, response["messages"][-1].content)])
            ),
            "nodes_calls": Annotated[set, operator.or_](
                set([("ml_dl_agent", (("text", response["messages"][-1].content),))])
            ),
        }
    )
