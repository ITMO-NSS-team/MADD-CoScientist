from langchain_core.output_parsers import PydanticOutputParser
from ChemCoScientist.agents.pydantic_models import Chat, Plan, Worker, Act, Translation

chat_parser = PydanticOutputParser(pydantic_object=Chat)
planner_parser = PydanticOutputParser(pydantic_object=Plan)
supervisor_parser = PydanticOutputParser(pydantic_object=Worker)
replanner_parser = PydanticOutputParser(pydantic_object=Act)
translator_parser = PydanticOutputParser(pydantic_object=Translation)