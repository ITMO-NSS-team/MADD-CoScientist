from ChemCoScientist.dataset_handler.chembl.chembl_utils import ChemblLoader
import yaml
from openai import OpenAI
import json
from ChemCoScientist.agents.agents_prompts import ds_builder_prompt, automl_prompt
from ChemCoScientist.tools import predict_prop_by_smiles, train_ml_with_data, get_state_from_server

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


from ChemCoScientist.agents.agents_prompts import worker_prompt, chat_prompt, supervisor_prompt, summary_prompt, replanner_prompt, planner_prompt, translate_prompt, retranslate_prompt
from ChemCoScientist.agents.parsers import chat_parser, supervisor_parser, replanner_parser, planner_parser, translator_parser
from ChemCoScientist.tools import chem_tools, web_tools, nanoparticle_tools
from ChemCoScientist.agents.pydantic_models import Response

import time


with open("./ChemCoScientist/conf/conf.yaml", "r") as file:
    conf = yaml.safe_load(file)
    key = conf["api_key"]
    base_url = conf["base_url"]
    file_path = conf["chembl_csv_path"]


def dataset_builder_agent(state):
    pending_tasks = state["pending_tasks"]
    responses = state["responses"]

    if not pending_tasks:
        return {"done": "validate", "responses": responses}

    user_query = pending_tasks.pop(0)
    prompt = ds_builder_prompt + user_query + r"""You: """

    chembl_client = ChemblLoader(True, file_path)
    llm_client = OpenAI(api_key=key, base_url=base_url)

    response = llm_client.chat.completions.create(
        model="meta-llama/llama-3.1-70b-instruct",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
    )
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
    
    
def ml_dl_agent(state):
    pending_tasks = state["pending_tasks"]
    responses = state["responses"]

    if not pending_tasks:
        return {"done": "validate", "responses": responses}

    user_query = pending_tasks.pop(0)
    prompt = automl_prompt 

    llm_client = ChatOpenAI(
        model="meta-llama/llama-3.1-70b-instruct",
        base_url=base_url,
        api_key=key,
        temperature=0.0,
        default_headers={"x-title": "DrugDesign"}
    )
    agent = create_react_agent(llm_client, [predict_prop_by_smiles, train_ml_with_data, get_state_from_server], state_modifier=prompt)
    
    response = agent.invoke({"messages": [("user", user_query)]})
    print("========")
    print(response['messages'][-1].content)
    print("========")

    if not pending_tasks:
        return {"done": "validate", "responses": response['messages'][-1].content}
    else:
        return {"done": False, "pending_tasks": pending_tasks, "responses": response['messages'][-1].content}


def in_translator_node(state, config: dict):
    """
    Processes an input string through a translation agent and determines the appropriate 
    response based on the detected language. Implements retry logic to handle API errors.

    Parameters
    ----------
    state : dict | TypedDict
        The current state of the system, expected to contain the key "input" with a string value.
    config : dict
        Configuration dictionary containing a "configurable" sub-dictionary with the LLM model 
        under the key "model".

    Returns
    -------
    Command
        A command object dictating the next state transition. If the detected language is English, 
        the command updates the state with 'language': 'English'. Otherwise, it includes both the 
        detected language and its translation. In case of critical API errors, an appropriate 
        termination response is returned.

    Raises
    ------
    Exception
        If an unhandled exception occurs, it is caught and retried up to `max_retries` times with 
        exponential backoff.

    Notes
    -----
    - The function uses an LLM pipeline consisting of `translate_prompt | llm | translator_parser`.
    - Implements exponential backoff (`1.2 ** attempt`) for error handling.
    - Handles API key errors and 404 errors explicitly.
    """


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
                    update= {'language': 'English'}
                )
            else:
                return Command(
                    goto='chat',
                    update={'language': output.language, 'translation': output.translation}
                )
            
        except Exception as e:  # Handle OpenAI API errors
            #logger.exception(f"InTranslator failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\n State: {state}")
            print(f"InTranslator failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            if 'api' in str(e).lower() and 'key' in str(e).lower():
                return Command(
                    goto=END,
                    update={"response": "Your api key is invalid"}
                )
            if '404' in str(e):
                return Command(
                    goto=END,
                    update={"response": "LLM is unavailabe right now, perhaps you should check your proxy"}
                )
            time.sleep(1.2**attempt)  # Exponential backoff
 

def re_translator_node(state, config: dict):
    """
    Re-translates a response into the specified language if it is not English.

    Parameters
    ----------
    state : dict | TypedDict
        The current execution state, containing "response" (the text to translate) 
        and "language" (the target language).
    config : dict
        Configuration dictionary containing a "configurable" sub-dictionary with the LLM model 
        under the key "model".

    Returns
    -------
    Command
        A command indicating the next state transition:
        - If the language is English, it returns the response unchanged.
        - Otherwise, it translates the response into the specified language.
        - If retries are exhausted, an error message is returned.

    Raises
    ------
    Exception
        Handles errors related to API failures, implementing exponential backoff (`2 ** attempt`).
    """
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
                update = {'response': output.content}
            )
            
        except Exception as e:  # Handle OpenAI API errors
            # logger.exception(f"ReTranslator failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\n State: {state}")
            print(f"ReTranslator failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(1.2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )

def chat_node(state, config: dict):
    """
    Processes user input through a chat agent and returns an appropriate response 
    or next action. This agent decides whether it can handle the user query itself. If yes, responds with the {"response": agent_answer}.
    Otherwise, calls main agentic system.

    Parameters
    ----------
    state : dict | TypedDict
        The current execution state, containing "input" (the user message) and 
        optionally "translation" if the language is not English.
    config : dict
        Configuration dictionary containing a "configurable" sub-dictionary with the LLM model 
        under the key "model".

    Returns
    -------
    dict
        If the response is a direct reply, returns {"response": message, "visualization": None}.
        If the response requires an action, returns {"next": action, "visualization": None}.
        If retries are exhausted, transitions to the planner with an empty response.

    Raises
    ------
    Exception
        Handles errors related to API failures, implementing exponential backoff (`2 ** attempt`).

    Notes
    -----
    - If the user's language is not English, it processes the translated text instead.
    - Resets visualization state on new responses.
    """
    llm: BaseChatModel = config["configurable"]["model"]
    chat_agent = chat_prompt | llm | chat_parser    
    input: str = state["input"] if state.get('language', 'English') == 'English' else state['translation']
    max_retries: int = 3


    for attempt in range(max_retries):
        try:
            output =  chat_agent.invoke(input)

            if isinstance(output.action, Response):
                return {"response": output.action.response, 'visualization': None} #we're setting visualization here to None to delete all previosly generated visualizatons
            else:
                return {"next": output.action.next, 'visualization': None}
            
        except Exception as e:  # Handle OpenAI API errors
            #logger.exception(f"Chat failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Chat failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(1.2**attempt)  # Exponential backoff

    return Command(
        goto='planner',
        update={"response": None}
    )

def should_end_chat(state):
    """
    Determines whether to continue the chat or transition to a different process.

    Parameters
    ----------
    state : dict | TypedDict
        The current execution state, expected to contain "response".

    Returns
    -------
    str
        Returns "retranslator" if a response exists, otherwise "planner".

    Notes
    -----
    - This function helps decide whether further processing is needed.
    """
    if "response" in state and state["response"]:
        return 'retranslator'
    else:
        return "planner"
    
def supervisor_node(state, config: dict):
    """
    Oversees the execution of a predefined plan by invoking an LLM-based supervisor.

    Parameters
    ----------
    state : dict | TypedDict
        The current execution state, expected to contain a "plan" (list of steps) 
        and optionally "input".
    config : dict
        Configuration dictionary containing a "configurable" sub-dictionary with the LLM model 
        under the key "model".

    Returns
    -------
    Command
        A command that either executes the next task in the plan or provides an 
        appropriate fallback message.

    Raises
    ------
    Exception
        Handles errors related to API failures, implementing exponential backoff (`2 ** attempt`).

    Notes
    -----
    - The function retrieves the first step from the plan and formats it for execution.
    - If no plan is available, it prompts the user to rephrase their question.
    """
    llm: BaseChatModel = config["configurable"]["model"]
    supervisor = supervisor_prompt | llm | supervisor_parser

    plan: list = state.get("plan")
    if plan is None and not state.get('input'):
        return Command(
            goto=END,
            update="I've couldn't answer to your question, could you ask me once more?-><-"
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
                update={"next": agent_response.next}
            )
        except Exception as e:  # Handle OpenAI API errors
            #logger.exception(f"Supervisor failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Supervisor failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(1.2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )

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
                          'nodes_calls': [('chemist_node', agent_response["messages"])]}
            )

        except Exception as e:  # Handle OpenAI API errors
            #logger.exception(f"Chemist failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Chemist failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(1.2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
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
                          'nodes_calls': [('nanoparticle_node', agent_response["messages"])]}
            )

        except Exception as e:  # Handle OpenAI API errors
            #logger.exception(f"Nanoparticle failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Nanoparticle failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(1.2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )

def automl_node(state, config: dict):
    """
    Executes an AutoML task by invoking an external Python script in a separate virtual environment.
    The function ensures that required dataset files are available before execution and handles 
    errors gracefully. It calls external 'fedot_llm' system with it's own llm_calls. By default it should use gpt4o-mini

    Parameters
    ----------
    state : dict | TypedDict
        The current execution state, including the planned steps and relevant contextual 
        information for the AutoML process.
    config : dict
        A dictionary containing configurable parameters, including the path to the dataset 
        directory and configurations for the AutoML system.

    Returns
    -------
    Command
        An object dictating the next step in execution ('replan' or `END`) and an update 
        to the execution state, recording results or errors encountered.

    Notes
    -----
    - The function calls an external AutoML script (`automl_invoke.py`) using a Python binary 
      from a specified virtual environment.
    - If the dataset directory is empty, the function aborts execution with a response message.
    - The subprocess call captures the script output, extracts a JSON response, and updates the state.
    - Handles errors related to JSON decoding and general execution failures.
    """
    #NOTE: here we call separate venv. Also automl_invoke.py uses state[input], not task

    script_path = "./automl_invoke.py"  # Adjust to actual location
    python_bin = "venv310/bin/python3.10"  # Ensure this points to Python 3.10 binary TODO: get it via environ

    plan: list = state["plan"]
    task: str = plan[0]

    dataset_dir_path = Path(config['configurable']['fedot_config']['user_data_dir']).resolve()
    if len(os.listdir(dataset_dir_path)) == 0:
        response_text = 'There is no files of dataset to use. Do not call me again'
        return Command(
            goto='replan',
            update={"past_steps": [(task, response_text)]}
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
                    'automl_results': response_text}
        )
    
    except json.JSONDecodeError as e:
        response_text = "I've couldn't do automl job, don't call me again"
        return Command(
            goto='replan',
            update={"past_steps": [(task, response_text)]}
        )
    except Exception as e:
        #logger.exception(f"automl failed with error: {str(e)}.\tState: {state}")
        return Command(
            goto=END,
            update={"response": f"I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
        )


def web_search_node(state, config: dict):
    """
    Executes a web search task using a language model (LLM) and predefined web tools.

    The function creates an agent to process a task from the execution plan. If web tools are 
    available, they are included in the agent's capabilities. The agent attempts to perform 
    the web search task and returns results, handling failures with retries.

    Parameters
    ----------
    state : dict | TypedDict
        The current execution state, including the plan with the task to execute.
    config : dict
        A dictionary containing configurable parameters, including the language model 
        and available web tools.

    Returns
    -------
    Command
        An object dictating the next step in execution ('replan' or `END`) and an update 
        to the execution state with results or errors.

    Notes
    -----
    - If no web tools are available, the function creates an agent without them.
    - The agent attempts the web search task up to three times, with exponential backoff on failure.
    - If the search fails after all retries, the function returns an error response.
    """
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
                          'nodes_calls': [('web_search_node', agent_response["messages"])]}
            )
        except Exception as e:  # Handle OpenAI API errors
            #logger.exception(f"Web_searcher failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Web_searcher failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(1.2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )


def plan_node(state, config):
    """
    Generates an execution plan using a language model (LLM) based on the provided input.

    The function processes an input query, determines a step-by-step execution plan, and 
    returns it. If an error occurs in parsing, it attempts to extract and correct the output.

    Parameters
    ----------
    state : dict | TypedDict
        The current execution state, containing the user input or its translated version.
    config : dict
        A dictionary containing configurable parameters, including the LLM model.

    Returns
    -------
    dict
        A dictionary with the generated plan (`"plan": list of steps`).
    Command
        If planning fails, returns a fallback response and terminates execution.

    Notes
    -----
    - Uses an LLM and a planner parser to generate structured steps.
    - Handles errors in JSON parsing and retries up to three times before failing.
    - Implements exponential backoff in case of API errors.
    """
    llm: BaseChatModel = config["configurable"]["model"]
    planner = planner_prompt | llm | planner_parser
    
    max_retries: int = 3
    input: str = state["input"] if state['language'] == 'English' else state['translation']

    for attempt in range(max_retries):
        try:
            plan = planner.invoke({"messages": [("user", input)]})
            return {"plan": plan.steps}
        except OutputParserException as  e:
            match = re.search(r'\{\s*"steps"\s*:\s*\[\s*(?:"[^"]*"\s*,\s*)*"[^"]*"\s*\]\s*\}', str(e), re.DOTALL)
            if match:
                json_str = match.group(0)
                json_str.replace("\\", "\\\\")
                try:
                    structured_output = json.loads(json_str)
                    return {"plan": structured_output['steps']}
                except json.JSONDecodeError as json_err:
                    #logger.exception(f"Planner failed with error: {str(json_err), json_str}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
                    print(f"Planner failed with error: {str(json_err)}. Retrying... ({attempt+1}/{max_retries})")
        except Exception as e:  # Handle OpenAI API errors
            #logger.exception(f"Planner failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Planner failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(1.2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )

def replan_node(state, config: dict):
    """
    Adjusts an existing execution plan based on past steps and the current state.
    This agent decides whether to respond with final answer or to continue executing tasks.
    This function refines or modifies a plan.

    Parameters
    ----------
    state : dict | TypedDict
        The current execution state, including the initial plan and past steps.
    config : dict
        A dictionary containing configurable parameters, including the LLM model.

    Returns
    -------
    dict
        A dictionary with the updated `"plan"` if replanning is successful.
        If a direct response is required, returns `"response"`.
    Command
        If replanning fails after retries, returns a fallback response.

    Notes
    -----
    - Uses an LLM and a replanner parser to adjust the execution plan.
    - Handles output parsing errors by extracting valid JSON structures.
    - Retries up to three times with exponential backoff before failing.
    """
    llm: BaseChatModel = config["configurable"]["model"]
    replanner = replanner_prompt | llm | replanner_parser

    input: str = state["input"] if state['language'] == 'English' else state['translation']
    max_retries: int = 3
    for attempt in range(max_retries):
        try:
            output =  replanner.invoke({'input': input, 'plan': state['plan'], 'past_steps': state['past_steps']})
            if isinstance(output.action, Response):
                return {"response": output.action.response}
            else:
                return {"plan": output.action.steps}
            
        except OutputParserException as  e:
            match = re.search(r'\{\s*"steps"\s*:\s*\[\s*(?:"[^"]*"\s*,\s*)*"[^"]*"\s*\]\s*\}', str(e), re.DOTALL)
            if match:
                json_str = match.group(0)
                json_str.replace("\\", "\\\\")
                try:
                    structured_output = json.loads(json_str)
                    return {"plan": structured_output['steps']}
                except json.JSONDecodeError as json_err:
                    #logger.exception(f"Planner failed with error: {str(json_err), json_str}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
                    print(f"RePlanner failed with error: {str(json_err)}. Retrying... ({attempt+1}/{max_retries})")
            
        except Exception as e:  # Handle OpenAI API errors
            #logger.exception(f"RePlanner failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"RePlanner failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(1.2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )

def should_end(state):
    """
    Determines the next step based on the presence of a response.

    This function decides whether execution should proceed to summarization
    or require further supervision.

    Parameters
    ----------
    state : PlanExecute
        The current execution state, potentially containing a generated response.

    Returns
    -------
    str
        `"summary"` if a response is available, otherwise `"supervisor"`.

    Notes
    -----
    - If the `"response"` key is present and non-empty, summarization is triggered.
    - If no response is available, the system proceeds to the supervisor node.
    """
    if "response" in state and state["response"]:
        return 'summary'
    else:
        return "supervisor"

def summary_node(state, config: dict):

    """
    Summarizes the system's response based on the provided input query and past steps using a summary agent. 

    This function attempts to invoke a summary agent to generate a summary based on the system's response, 
    the query, and past steps. It retries up to a maximum number of attempts if an error occurs during the 
    process, with exponential backoff for each retry.

    Parameters
    ----------
    state : dict | TypedDict
        The current state containing details about the system's response, input query, past steps, and other context.
        The dictionary should include:
        - "response" : The system's response to the previous query.
        - "input" : The query input.
        - "language" : Optional, language of the query (default is 'English').
        - "translation" : Optional, translation of the query if language is not 'English'.
        - "past_steps" : A list of previous steps or intermediate thoughts.
    
    config : dict
        The configuration dictionary that contains settings such as the model to be used for generating summaries.
        This should include:
        - "configurable" : A dictionary with "model" key, specifying the LLM (Language Model) used for generating summaries.

    Returns
    -------
    dict
        A dictionary with the summarized response as 'response'. If all retry attempts fail, returns a command to go to 
        the end state with an error message.

    Notes
    -----
    If the summary generation fails after the maximum number of retries, the function will return a response indicating
    that the summary could not be generated and prompts for further help.
    """

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
            return {'response': output.content}
            
        except Exception as e:  # Handle OpenAI API errors
            #logger.exception(f"Summary failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\n State: {state}")
            print(f"Summary failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(1.2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )