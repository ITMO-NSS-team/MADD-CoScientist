from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent

import subprocess
import json
import re   
import os
from pathlib import Path

from graph.states import PlanExecute
from prompts import worker_prompt, memory_prompt, chat_prompt, chat_parser, supervisor_prompt, supervisor_parser, summary_prompt, replanner_prompt, replanner_parser, planner_parser, planner_prompt, translate_prompt, translator_parser, retranslate_prompt
from tools import chem_tools, web_tools, nanoparticle_tools
from pydantic_models import Response
from ChemCoScientist.memory import AgentMemory

import time
from typing import List
import logging

# Create a separate logger for nodes.py
logger = logging.getLogger("node_logger")
logger.setLevel(logging.INFO)

# Configure a file handler for the node logger
file_handler = logging.FileHandler("node.log")
file_handler.setLevel(logging.INFO)

# Set a formatter for the node logger
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the node logger
logger.addHandler(file_handler)




def in_translator_node(state: PlanExecute, config: dict):

    llm: BaseChatModel = config["configurable"]["model"]
    translator_agent = translate_prompt | llm | translator_parser


    input: str = state["input"]
    max_retries: int = 3


    for attempt in range(max_retries):
        try:
            output =  translator_agent.invoke(input)

            if output.language =='English':
                return Command(
                    goto = 'chat',
                    update= {'language': 'English', 'agent_memory': state["agent_memory"]}
                )
            else:
                return Command(
                    goto='chat',
                    update={'language': output.language, 'translation': output.translation, 'agent_memory': state["agent_memory"]}
                )
            
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"InTranslator failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\n State: {state}")
            print(f"InTranslator failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            if 'api' in str(e).lower() and 'key' in str(e).lower():
                return Command(
                    goto=END,
                    update={"response": "Your api key is invalid", 'agent_memory': state["agent_memory"]}
                )
            if '404' in str(e):
                return Command(
                    goto=END,
                    update={"response": "LLM is unavailabe right now, perhaps you should check your proxy", 'agent_memory': state["agent_memory"]}
                )
            time.sleep(2**attempt)  # Exponential backoff
    

def re_translator_node(state: PlanExecute, config: dict):
    input: str = state["response"]
    language: str = state['language']

    llm: BaseChatModel = config["configurable"]["model"]
    #memorize = memory_prompt | llm
    max_retries: int = 3

    if language == 'English':
        return Command(
            goto=END,
            update = {'response': input}
        )

    translator_agent = retranslate_prompt | llm 

    for attempt in range(max_retries):
        try:
            output =  translator_agent.invoke({"input": input, "language": language})
                return Command(
                    goto=END,
                    update = {'response': output.content, 'agent_memory': state["agent_memory"]}
                )
            
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"ReTranslator failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\n State: {state}")
            print(f"ReTranslator failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-", 'agent_memory': state["agent_memory"]}
    )


def chat_node(state: PlanExecute, config: dict):

    llm: BaseChatModel = config["configurable"]["model"]
    chat_agent = chat_prompt | llm | chat_parser    
    input: str = state["input"] if state.get('language', 'English') == 'English' else state['translation']
    max_retries: int = 3


    for attempt in range(max_retries):
        try:
            output =  chat_agent.invoke(input)

            if isinstance(output.action, Response):
                return {"response": output.action.response, 'visualization': None, 'agent_memory': state["agent_memory"]} #we're setting visualization here to None to delete all previosly generated visualizatons
            else:
                return {"next": output.action.next, 'visualization': None, 'agent_memory': state["agent_memory"]}
            
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"Chat failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Chat failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto='planner',
        update={"response": None, 'agent_memory': state["agent_memory"]}
    )

def should_end_chat(state: PlanExecute):
    if "response" in state and state["response"]:
        return 'retranslator'
    else:
        return "planner"


def supervisor_node(state: PlanExecute, config: dict):

    llm: BaseChatModel = config["configurable"]["model"]
    supervisor = supervisor_prompt | llm | supervisor_parser


    plan: list = state.get("plan")
    if plan is None and not state.get('input'):
        return Command(
            goto=END,
            update={"response": "I've couldn't answer to your question, could you ask me once more?-><-", 'agent_memory': state["agent_memory"]}
        )
    plan_str: str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    task: str = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""
    
    max_retries: int = 3


    for attempt in range(max_retries):
        try:
            agent_response = supervisor.invoke({"input": [("user", task_formatted)]})

            return Command(
                goto=agent_response.next,
                update={"next": agent_response.next, 'agent_memory': state["agent_memory"]}
            )
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"Supervisor failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Supervisor failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-", 'agent_memory': state["agent_memory"]}
    )



def chemist_node(state: PlanExecute, config: dict):

    llm: BaseChatModel = config["configurable"]["model"]
    chem_agent = create_react_agent(llm, chem_tools, state_modifier=worker_prompt + 'admet = qed')

    plan: list = state["plan"]
    plan_str: str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    task: str = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            config['configurable']['state'] = state #to get tool_call_id, but this shouldn't be implemented like that
            agent_response = chem_agent.invoke({"messages": [("user", task_formatted)]})

            return Command(
                goto = 'replan',
                update = {'past_steps':[(task, agent_response["messages"][-1].content)],
                          'nodes_calls': [('chemist_node', agent_response["messages"])],
                          'agent_memory': state["agent_memory"]}
            )

        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"Chemist failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Chemist failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-", 'agent_memory': state["agent_memory"]}
    )

def nanoparticle_node(state: PlanExecute, config: dict):

    llm: BaseChatModel = config["configurable"]["model"]
    #add_prompt = 'if you are asked to predict nanoparticle shape, directly call corresponding tool'
    add_prompt = 'You have to respond with results of tool call, do not repharse it'
    nanoparticle_agent = create_react_agent(llm, nanoparticle_tools, state_modifier=worker_prompt + add_prompt)

    plan: list = state["plan"]
    plan_str: str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    task: str = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            agent_response = nanoparticle_agent.invoke({"messages": [("user", task_formatted)]})

            return Command(
                goto = 'replan',
                update = {'past_steps':[(task, agent_response["messages"][-1].content)],
                          'nodes_calls': [('nanoparticle_node', agent_response["messages"])],
                          'agent_memory': state["agent_memory"]}
            )

        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"Nanoparticle failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Nanoparticle failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-", 'agent_memory': state["agent_memory"]}
    )
    
def automl_node(state: PlanExecute, config: dict):
    #NOTE: here we call separate venv. Also automl_invoke.py uses state[input], not task

    script_path = "graph/automl_invoke.py"  # Adjust to actual location
    python_bin = "venv310/bin/python3.10"  # Ensure this points to Python 3.10 binary TODO: get it via environ

    plan: list = state["plan"]
    task: str = plan[0]

    dataset_dir_path = Path(config['configurable']['fedot_config']['user_data_dir']).resolve()
    if len(os.listdir(dataset_dir_path)) == 0:
        response_text = 'There is no files of dataset to use. Do not call me again'
        return Command(
            goto='replan',
            update={"past_steps": [(task, response_text)], 'agent_memory': state["agent_memory"]}
        )
    
    try:
        result = subprocess.run(
            [python_bin, script_path, json.dumps(config['configurable']['fedot_config']), json.dumps(state)],
            capture_output=True, text=True
        )

        re_match = re.search(r'\{.*\}', result.stdout, re.DOTALL)
        response_text = json.loads(re_match.group(0).encode('utf-16', 'surrogatepass').decode('utf-16'))['response']

        #response_text = 'I have done automl job, results are saved'
        return Command(
            goto='replan',
            update={"past_steps": [(task, response_text)],
                    'automl_results': response_text,
                    'agent_memory': state["agent_memory"]}
        )
    
    except json.JSONDecodeError as e:
        response_text = "I've couldn't do automl job, don't call me again"
        return Command(
            goto='replan',
            update={"past_steps": [(task, response_text)], 'agent_memory': state["agent_memory"]}
        )
    except Exception as e:
        logger.exception(f"automl failed with error: {str(e)}.\tState: {state}")
        return Command(
            goto=END,
            update={"response": f"I can't answer to your question right now( Perhaps there is something else that I can help? -><-", 'agent_memory': state["agent_memory"]}
        )

def web_search_node(state: PlanExecute, config: dict):

    llm: BaseChatModel = config["configurable"]["model"]
    if not web_tools:
        web_agent = create_react_agent(llm, [], state_modifier=worker_prompt)
    else:
        web_agent = create_react_agent(llm, web_tools, state_modifier=worker_prompt)

    plan: list = state["plan"]
    plan_str: str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    task: str = plan[0]
    task_formatted: str = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""
    
    max_retries: int = 3
    for attempt in range(max_retries):
        try:
            agent_response = web_agent.invoke({"messages": [("user", task_formatted)]})
            return Command(
                goto = 'replan',
                update = {'past_steps':[(task, agent_response["messages"][-1].content)],
                          'nodes_calls': [('web_search_node', agent_response["messages"])],
                          'agent_memory': state["agent_memory"]}
            )
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"Web_searcher failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Web_searcher failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-", 'agent_memory': state["agent_memory"]}
    )

def plan_node(state: PlanExecute, config):

    llm: BaseChatModel = config["configurable"]["model"]
    planner = planner_prompt | llm | planner_parser
    
    max_retries: int = 3
    input: str = state["input"] if state['language'] == 'English' else state['translation']

    for attempt in range(max_retries):
        try:
            plan = planner.invoke({"messages": [("user", input)]})
            return {"plan": plan.steps, 'agent_memory': state["agent_memory"]}
        except OutputParserException as  e:
            match = re.search(r'\{\s*"steps"\s*:\s*\[\s*(?:"[^"]*"\s*,\s*)*"[^"]*"\s*\]\s*\}', str(e), re.DOTALL)
            if match:
                json_str = match.group(0)
                json_str.replace("\\", "\\\\")
                try:
                    structured_output = json.loads(json_str)
                    return {"plan": structured_output['steps'], 'agent_memory': state["agent_memory"]}
                except json.JSONDecodeError as json_err:
                    logger.exception(f"Planner failed with error: {str(json_err), json_str}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
                    print(f"Planner failed with error: {str(json_err)}. Retrying... ({attempt+1}/{max_retries})")
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"Planner failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Planner failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-", 'agent_memory': state["agent_memory"]}
    )

def replan_node(state: PlanExecute, config: dict):

    llm: BaseChatModel = config["configurable"]["model"]
    replanner = replanner_prompt | llm | replanner_parser

    input: str = state["input"] if state['language'] == 'English' else state['translation']
    max_retries: int = 3
    for attempt in range(max_retries):
        try:
            output =  replanner.invoke({'input': input, 'plan': state['plan'], 'past_steps': state['past_steps']})
            if isinstance(output.action, Response):
                return {"response": output.action.response, 'agent_memory': state["agent_memory"]}
            else:
                return {"plan": output.action.steps, 'agent_memory': state["agent_memory"]}
            
        except OutputParserException as  e:
            match = re.search(r'\{\s*"steps"\s*:\s*\[\s*(?:"[^"]*"\s*,\s*)*"[^"]*"\s*\]\s*\}', str(e), re.DOTALL)
            if match:
                json_str = match.group(0)
                json_str.replace("\\", "\\\\")
                try:
                    structured_output = json.loads(json_str)
                    return {"plan": structured_output['steps'], 'agent_memory': state["agent_memory"]}
                except json.JSONDecodeError as json_err:
                    logger.exception(f"Planner failed with error: {str(json_err), json_str}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
                    print(f"RePlanner failed with error: {str(json_err)}. Retrying... ({attempt+1}/{max_retries})")
            
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"RePlanner failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"RePlanner failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-", 'agent_memory': state["agent_memory"]}
    )

def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return 'summary'
    else:
        return "supervisor"
    
def summary_node(state: PlanExecute, config: dict):
    system_response: str = state["response"]
    query: str = state["input"] if state.get('language', 'English') == 'English' else state['translation']
    past_steps: list = state["past_steps"]
    llm: BaseChatModel = config["configurable"]["model"]
    #memorize = memory_prompt | llm
    max_retries: int = 3

    summary_agent = summary_prompt | llm 

    for attempt in range(max_retries):
        try:
            output =  summary_agent.invoke({'query': query, 'system_response': system_response, 'intermediate_thoughts': past_steps})
            return {'response': output.content, 'agent_memory': state["agent_memory"]}
            
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"Summary failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\n State: {state}")
            print(f"Summary failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-", 'agent_memory': state["agent_memory"]}
    )can't answer to your question right now( Perhaps there is something else that I can help? -><-", 'agent_memory': state["agent_memory"]}