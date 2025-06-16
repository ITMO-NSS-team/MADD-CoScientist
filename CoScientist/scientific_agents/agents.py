from smolagents import CodeAgent, OpenAIServerModel
from smolagents import DuckDuckGoSearchTool

from langgraph.types import Command
from ChemCoScientist.memory import AgentMemory


def coder_agent(state: dict, config: dict):
    config_cur_agent = config["configurable"]["additional_agents_info"]["coder_agent"]
    memory = AgentMemory(state)
    plan = state["plan"]
    task = plan[0]

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

    return Command(
        goto="replan_node",
        update={
            "past_steps": [(task, str(response))],
            "nodes_calls": [("coder_agent", str(response))],
            "agent_memory": state["agent_memory"],
        },
    )   