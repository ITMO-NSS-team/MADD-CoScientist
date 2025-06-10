from smolagents import CodeAgent, OpenAIServerModel
from smolagents import DuckDuckGoSearchTool
from ChemCoScientist.tools.chemist_tools import fetch_chembl_data, fetch_BindingDB_data


search_tool = DuckDuckGoSearchTool()

model = OpenAIServerModel(api_base="https://api.vsegpt.ru/v1", model_id="deepseek/deepseek-chat-0324-alt-structured", api_key="KEY")
agent = CodeAgent(tools=[search_tool, fetch_BindingDB_data, fetch_chembl_data], model=model, additional_authorized_imports=['*', 'joblib', 'requests', 'pickle', 'sklearn', 'pubchempy', 'scikit-learn', 'numpy', 'pandas', 'os', 'bs4', 'smolagents', 'rdkit', 'rdkit.Chem', 'rdkit.Chem.Descriptors', 'rdkit.Chem.Draw', 'xlrd'])
dir = "/Users/alina/Desktop/ИТМО/ChemCoScientist/ChemCoScientist/data_dir_for_coder"
main_prompt = f"To generate code you have access to libraries: 're', 'rdkit', \
'smolagents', 'math', 'stat', 'datetime', 'os', 'time', 'requests', 'queue', \
'random', 'bs4', 'rdkit.Chem', 'unicodedata', 'itertools', 'statistics', 'pubchempy',\
'rdkit.Chem.Draw', 'collections', 'numpy', 'rdkit.Chem.Descriptors', 'sklearn', 'pickle', 'joblib'. \
You are an agent who helps prepare a chemical dataset. \
You can download data from ChemBL, BindingDB and process it. In your answers you must say the full path to the file. You ALWAYS save all results in excel tables.\
Attention!!! Directory for saving files: {dir}.\
AFTER the answer you should express your opinion whether this data is enough to train the ml-model!!!"

additional_prompt = "\n Хватит ли данных для обучения модели? Напиши путь, куда сохранила."
    
    
if __name__ =="__main__":
    user_input = "Получи данные Ki по Q9BPZ7 из BindingDB."
    agent.run(main_prompt + '\n' + user_input + additional_prompt)