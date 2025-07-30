import os
import time
from typing import Annotated
import operator

from langgraph.types import Command
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent
from smolagents import CodeAgent, LiteLLMModel, OpenAIServerModel

from ChemCoScientist.agents.agents_prompts import (
    additional_ds_builder_prompt,
    automl_prompt,
    ds_builder_prompt,
    worker_prompt,
)
from ChemCoScientist.paper_analysis.question_processing import process_question
from ChemCoScientist.tools import chem_tools, nanoparticle_tools
from ChemCoScientist.tools.chemist_tools import fetch_BindingDB_data, fetch_chembl_data
from ChemCoScientist.tools.ml_tools import agents_tools as automl_tools


def get_all_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


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


def chemist_node(state: dict, config: dict) -> Command:
    print("--------------------------------")
    print("Chemist agent called")
    print("Current task:")
    print(state["task"])
    print("--------------------------------")

    task = state["task"]
    plan = state["plan"]
    llm = config["configurable"]["llm"]

    chem_agent = create_react_agent(
        llm, chem_tools, state_modifier=worker_prompt + "admet = qed"
    )

    task_formatted = f"""For the following plan:\n{str(plan)}\n\nYou are tasked with executing: {task}."""

    for attempt in range(3):
        try:
            config["configurable"]["state"] = state
            agent_response = chem_agent.invoke({"messages": [("user", task_formatted)]})

            return Command(update={
                "past_steps": Annotated[set, operator.or_](set([
                    (task, agent_response["messages"][-1].content)
                ])),
                "nodes_calls": Annotated[set, operator.or_](set([
                    (
                        "chemist_node",
                        tuple((m.type, m.content) for m in agent_response["messages"])
                    )
                ])),
            })

        except Exception as e:
            print(f"Chemist failed: {str(e)}. Retrying ({attempt+1}/3)")
            time.sleep(1.2**attempt)

    return Command(goto=END, update={
        "response": "I can't answer your question right now. Perhaps I can help with something else?"
    })


def nanoparticle_node(state: dict, config: dict) -> Command:
    print("--------------------------------")
    print("Nano-p agent called")
    print("Current task:")
    print(state["task"])
    print("--------------------------------")

    task = state["task"]
    plan = state["plan"]
    llm = config["configurable"]["llm"]

    nanoparticle_agent = create_react_agent(
        llm, nanoparticle_tools,
        state_modifier=worker_prompt + "You have to respond with results of tool call, do not rephrase it"
    )

    task_formatted = f"""For the following plan:\n{str(plan)}\n\nYou are tasked with executing: {task}."""

    for attempt in range(3):
        try:
            agent_response = nanoparticle_agent.invoke({"messages": [("user", task_formatted)]})

            return Command(update={
                "past_steps": Annotated[set, operator.or_](set([
                    (task, agent_response["messages"][-1].content)
                ])),
                "nodes_calls": Annotated[set, operator.or_](set([
                    (
                        "nanoparticle_node",
                        tuple((m.type, m.content) for m in agent_response["messages"])
                    )
                ])),
            })

        except Exception as e:
            print(f"Nanoparticle error: {str(e)}. Retrying ({attempt+1}/3)")
            time.sleep(1.2**attempt)

    return Command(goto=END, update={
        "response": "I can't answer your question right now. Perhaps I can help with something else?"
    })


def paper_analysis_node(state: dict, config: dict) -> Command:
    print("--------------------------------")
    print("Paper agent called")
    print("Current task:")
    print(state["task"])
    print("--------------------------------")

    task = state["task"]
    response = process_question(task)

    updated_metadata = state.get("metadata", {}).copy()
    updated_metadata["paper_analysis"] = response.get("metadata")
    updated_metadata = {}
    response = {'answer': 'Paper agent not supported now'}
    
    return Command(update={
        "past_steps": Annotated[set, operator.or_](set([
            (task, response.get("answer"))
        ])),
        "nodes_calls": Annotated[set, operator.or_](set([
            ("paper_analysis_node", (("text", response.get("answer")),))
        ])),
        "metadata": Annotated[dict, operator.or_](updated_metadata),
    })
