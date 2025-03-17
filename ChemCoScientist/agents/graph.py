import logging
import os

from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from ChemCoScientist.agents import (chat_node, chemist_node, in_translator_node,
                         nanoparticle_node, plan_node, re_translator_node,
                         replan_node, should_end, should_end_chat,
                         summary_node, supervisor_node, web_search_node,
                         automl_node)

from ChemCoScientist.agents.states import PlanExecute

# # Create a separate logger for nodes.py
# logger = logging.getLogger("graph_logger")
# logger.setLevel(logging.INFO)

# # Configure a file handler for the node logger
# file_handler = logging.FileHandler("graph.log")
# file_handler.setLevel(logging.INFO)

# # Set a formatter for the node logger
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)

# # Add the handler to the node logger
# logger.addHandler(file_handler)


workflow = StateGraph(PlanExecute)

# Add the in_translator node
workflow.add_node("intranslator", in_translator_node)
# Add the chat node
workflow.add_node("chat", chat_node)
# Add the plan node
workflow.add_node("planner", plan_node)
# Add the supervisor node
workflow.add_node("supervisor", supervisor_node)
# Add the chemist step
workflow.add_node("chemist", chemist_node)
# Add the nanoparticle step
workflow.add_node("nanoparticles", nanoparticle_node)
# Add the web_search step
workflow.add_node("web_search", web_search_node)
# Add a replan node
workflow.add_node("replan", replan_node)
# Add the out_translator node
workflow.add_node("summary", summary_node)
# Add the out_translator node
workflow.add_node("retranslator", re_translator_node)
workflow.add_node("automl", automl_node)


workflow.add_edge(START, "intranslator")
workflow.add_conditional_edges(
    "chat",
    # Next, we pass in the function that will determine which node is called next.
    should_end_chat,
    ["planner", 'retranslator'],
)
# From plan we go to supervisor
workflow.add_edge("planner", "supervisor")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["supervisor", 'summary'],
)
workflow.add_edge("summary", "retranslator")
# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

class App:
    def __init__(self, main_model_name: str, visual_model_name: str, fedot_model_name: str,  base_url: str, api_key: str, tavily_api_key:str = None):
        self.llm = ChatOpenAI(model=main_model_name,
                        base_url=base_url,
                        api_key=api_key,
                        temperature=0.7,
                        default_headers={"x-title": "ChemicalChatBot"})

        self.visual_llm = ChatOpenAI(model=visual_model_name,
                        base_url=base_url,
                        api_key=api_key,
                        temperature=1.0,
                        default_headers={"x-title": "ChemicalChatBot"})
        
        self.fedot_model = fedot_model_name
        self.api_key = api_key
        self.base_url = base_url

        if tavily_api_key:
            os.environ['TAVILY_API_KEY'] = tavily_api_key
        self.app = app

    def invoke(self, input: dict, config: RunnableConfig):
        config['configurable']['model'] = self.llm
        config['configurable']['visual_model'] = self.visual_llm

        user_data_dir = config['configurable'].get('user_data_dir')
        fedot_config = {
                        'user_data_dir': user_data_dir,
                        "model_name": self.fedot_model,
                        'openai_api_base': self.base_url,
                        'openai_api_key': self.api_key
                        }
        
        config['configurable']['fedot_config'] = fedot_config
        
        #logger.info(f"\n\nINPUT: {input}")
        for event in app.stream(input=input, config=config):
            for k, v in event.items():
                if k != "__end__":
                    pass
                   #logger.info(v)
        #return self.app.invoke(input=input, config=config)
        return v
    
    def stream(self, input: dict, config: RunnableConfig):
        config['configurable']['model'] = self.llm
        config['configurable']['visual_model'] = self.visual_llm
        
        user_data_dir = config['configurable'].get('user_data_dir')
        fedot_config = {
                        'user_data_dir': user_data_dir,
                        "model_name": self.fedot_model,
                        'openai_api_base': self.base_url,
                        'openai_api_key': self.api_key
                        }
        
        config['configurable']['fedot_config'] = fedot_config
        #logger.info(f"\n\nINPUT: {input}")
        for event in app.stream(input=input, config=config):
            for k, v in event.items():
                if k != "__end__":
                   #logger.info(v)
                   pass
                yield v

    async def ainvoke(self, input: dict, config: RunnableConfig):
        config['configurable']['model'] = self.llm
        config['configurable']['visual_model'] = self.visual_llm
        return self.app.ainvoke(input=input, config=config)

    async def astream(self, input: dict, config: RunnableConfig):
        config['configurable']['model'] = self.llm
        config['configurable']['visual_model'] = self.visual_llm
        return self.app.astream(input=input, config=config)

