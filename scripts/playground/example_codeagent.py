from smolagents import CodeAgent, OpenAIServerModel
from smolagents import DuckDuckGoSearchTool
from ChemCoScientist.tools.chemist_tools import fetch_chembl_data


search_tool = DuckDuckGoSearchTool()
dir = "/Users/alina/Desktop/ИТМО/ChemCoScientist/"
model = OpenAIServerModel(api_base="https://api.vsegpt.ru/v1", model_id="deepseek/deepseek-chat-0324-alt-structured", api_key="KEY")
agent = CodeAgent(tools=[search_tool, fetch_chembl_data], model=model, additional_authorized_imports=['*'])
main_prompt = "To generate code you have access to libraries: 're', 'rdkit', \
'smolagents', 'math', 'stat', 'datetime', 'os', 'time', 'requests', 'queue', \
'random', 'bs4', 'rdkit.Chem', 'unicodedata', 'itertools', 'statistics', 'pubchempy',\
'rdkit.Chem.Draw', 'collections', 'numpy', 'rdkit.Chem.Descriptors', 'sklearn', 'pickle', 'joblib'. \
Attention!!! Directory for saving files: " + dir
# agent.run("Нарисуй молекулу и сохрани в '/Users/alina/Desktop/ИТМО/ChemCoScientist/', молекула 'Cc1ccccc1'"+ main_prompt)
# agent.run("Сделай проверку валидности реакции для [C:1]-O-C>>[C:1]-O" + main_prompt)
# agent.run("Напиши молекулы, что есть в файле /Users/alina/Desktop/ИТМО/ChemCoScientist/answers_no_sum_<12.xlsx?" + main_prompt)
# agent.run("Для каждой молекулы из /Users/alina/Desktop/ИТМО/ChemCoScientist/filtered_data.csv посчитай RO5 и создай в этой же директории файл с этими данными." + main_prompt)
# agent.run("Напиши значения RO5 для первой молекулы из /Users/alina/Desktop/ИТМО/ChemCoScientist/filtered_data_with_ro5.csv." + main_prompt)
# agent.run("Проходит ли фильтр RO5 для первой молекулы из /Users/alina/Desktop/ИТМО/ChemCoScientist/filtered_data_with_ro5.csv?" + main_prompt)

while True:
    user_input = input()
    agent.run(main_prompt + '\n' + user_input)