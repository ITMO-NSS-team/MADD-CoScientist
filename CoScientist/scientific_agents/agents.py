from smolagents import CodeAgent, OpenAIServerModel
from smolagents import DuckDuckGoSearchTool
from smolagents import LiteLLMModel

from langgraph.types import Command
from typing import Annotated


def coder_agent(state: dict, config: dict):
    config_cur_agent = config["configurable"]["additional_agents_info"]["coder_agent"]
    plan = state["plan"]
    task = plan[0]

    if 'groq.com' in config_cur_agent["url"]:
        model = LiteLLMModel(
            config_cur_agent["model_name"],
            api_base=config_cur_agent["url"],
            api_key=config_cur_agent["api_key"]
        )
    else:
        model = OpenAIServerModel(
            api_base=config_cur_agent["url"],
            model_id=config_cur_agent["model_name"],
            api_key=config_cur_agent["api_key"],
        )
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool()], model=model, additional_authorized_imports=["*"]
    )
    main_prompt = (
        "To generate code you have access to libraries: 're', 'rdkit', \
    'smolagents', 'math', 'stat', 'datetime', 'os', 'time', 'requests', 'queue', \
    'random', 'bs4', 'rdkit.Chem', 'unicodedata', 'itertools', 'statistics', 'pubchempy',\
    'rdkit.Chem.Draw', 'collections', 'numpy', 'rdkit.Chem.Descriptors', 'sklearn', 'pickle', 'joblib'. \
    Attention!!! Directory for saving files: "
        + config_cur_agent["ds_dir"]
    )
    response = agent.run(main_prompt + "\n" + task)

    return Command(update={
        "nodes_calls": Annotated[list, "accumulate"]([("coder_agent", str(response))]),
        "past_steps": Annotated[list, "accumulate"]([(task, str(response))]),
    })
