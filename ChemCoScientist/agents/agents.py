import json
import os
import re
import subprocess
import time
from pathlib import Path

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from protollm.agents.agent_utils.pydantic_models import Response

from ChemCoScientist.agents.agents_prompts import (automl_prompt,
                                                   ds_builder_prompt,
                                                   worker_prompt)
from ChemCoScientist.dataset_handler.chembl.chembl_utils import ChemblLoader
from ChemCoScientist.tools import (chem_tools, get_state_from_server,
                                   nanoparticle_tools, predict_prop_by_smiles,
                                   train_ml_with_data)

with open("./ChemCoScientist/conf/conf.yaml", "r") as file:
    conf = yaml.safe_load(file)
    key = conf["api_key"]
    base_url = conf["base_url"]
    file_path = conf["chembl_csv_path"]


def dataset_builder_agent(state, config: dict):
    input = (
        state["input"]
        if state.get("language", "English") == "English"
        else state["translation"]
    )
    llm = config["configurable"]["model"]
    chembl_client = ChemblLoader(True, file_path)

    prompt = ds_builder_prompt + user_query + r"""You: """

    response = llm.invoke([{"role": "system", "content": prompt}])
    print("========")
    print(response.choices[0].message.content)
    print("========")

    try:
        query_params = eval(response.choices[0].message.content)
    except:
        try:
            query_params = eval(response.choices[0].message.content.split("You: ")[-1])
        except:
            return {
                "done": "error",
                "message": "Failed to parse LLM response.",
                "responses": None,
            }

    selected_columns = query_params.get("selected_columns", [])
    filters = query_params.get("filters", {})

    result_df = chembl_client.get_filtered_data(selected_columns, filters)
    responses.append(result_df)

    if not pending_tasks:
        return {"done": "validate", "responses": responses}
    else:
        return {"done": False, "pending_tasks": pending_tasks, "responses": responses}


def ml_dl_agent(state, config: dict):
    user_query = (
        state["input"]
        if state.get("language", "English") == "English"
        else state["translation"]
    )
    prompt = automl_prompt

    plan: list = state["plan"]
    plan_str: str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task: str = plan[0]

    llm: BaseChatModel = config["configurable"]["llm"]
    agent = create_react_agent(
        llm,
        [predict_prop_by_smiles, train_ml_with_data, get_state_from_server],
        state_modifier=prompt,
    )

    agent_response = agent.invoke({"messages": [("user", user_query)]})

    return Command(
        goto="replan_node",
        update={
            "past_steps": [(task, agent_response["messages"][-1].content)],
            "nodes_calls": [("nanoparticle_node", agent_response["messages"])],
        },
    )
    print("========")
    print(response["messages"][-1].content)
    print("========")


def chemist_node(state, config: dict):
    """
    Executes a chemistry-related task using a ReAct-based agent.

    Parameters
    ----------
    state : dict | TypedDict
        The current execution state containing the task plan.
    config : dict
        Configuration dictionary containing the LLM model and related settings.

    Returns
    -------
    Command
        An object specifying the next execution step and updates to the state.
    """
    llm: BaseChatModel = config["configurable"]["llm"]
    chem_agent = create_react_agent(
        llm, chem_tools, state_modifier=worker_prompt + "admet = qed"
    )

    plan: list = state["plan"]
    plan_str: str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    task: str = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            config["configurable"][
                "state"
            ] = state  # to get tool_call_id, but this shouldn't be implemented like that
            agent_response = chem_agent.invoke({"messages": [("user", task_formatted)]})

            return Command(
                goto="replan_node",
                update={
                    "past_steps": [(task, agent_response["messages"][-1].content)],
                    "nodes_calls": [("chemist_node", agent_response["messages"])],
                },
            )

        except Exception as e:  # Handle OpenAI API errors
            # logger.exception(f"Chemist failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(
                f"Chemist failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})"
            )
            time.sleep(1.2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={
            "response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"
        },
    )


def nanoparticle_node(state, config: dict):
    """
    Executes a task related to nanoparticle analysis using a large language model (LLM)
    and associated tools, retrying up to a maximum number of times in case of failure.

    Parameters
    ----------
    state : dict | TypedDict
        The current execution state, which includes the planned steps and other contextual
        information for processing.
    config : dict
        A dictionary containing configurable parameters, including the LLM model
        and any relevant tools required for task execution.

    Returns
    -------
    Command
        An object containing the next node to transition to ('replan' or `END`) and
        an update to the execution state with recorded steps and responses.

    Notes
    -----
    - The function constructs a task prompt based on the execution plan and invokes
      an LLM-based agent to process the task.
    - If an exception occurs, it retries the operation with exponential backoff.
    - If all retries fail, it returns a response indicating failure.
    """
    llm: BaseChatModel = config["configurable"]["llm"]
    # add_prompt = 'if you are asked to predict nanoparticle shape, directly call corresponding tool'
    add_prompt = "You have to respond with results of tool call, do not repharse it"
    nanoparticle_agent = create_react_agent(
        llm, nanoparticle_tools, state_modifier=worker_prompt + add_prompt
    )

    plan: list = state["plan"]
    plan_str: str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    task: str = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            agent_response = nanoparticle_agent.invoke(
                {"messages": [("user", task_formatted)]}
            )

            return Command(
                goto="replan_node",
                update={
                    "past_steps": [(task, agent_response["messages"][-1].content)],
                    "nodes_calls": [("nanoparticle_node", agent_response["messages"])],
                },
            )

        except Exception as e:  # Handle OpenAI API errors
            # logger.exception(f"Nanoparticle failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(
                f"Nanoparticle failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})"
            )
            time.sleep(1.2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={
            "response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"
        },
    )
