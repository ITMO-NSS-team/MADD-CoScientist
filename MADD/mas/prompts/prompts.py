import os

from langchain_core.prompts import ChatPromptTemplate

ds_builder_prompt = f"You can download data from ChemBL, BindingDB. \n\
Rules: \n\
1) Don't call downloading from ChemBL, BindingDB unless they ask you to download! \n\
2) Never invent IDs from the database yourself. Specify them only if the user names them himself.\n\
3) Don't change the protein name from the user's request. If they ask for SARS-CoV-2, then pass the protein_name unchanged.\n\
"

automl_prompt = f"""You have access to two types of generative models and to tools for predict properties, run training, check status of training:

        1. PRE-DEFINED DISEASE MODELS (always available):
        - Alzheimer's disease → use 'generate_mol_by_case' with arg 'Alzheimer'. Description of disease: GSK-3beta inhibitors with high activity. \
    These molecules can bind to GSK-3beta protein, molecules has low brain-blood barrier permeability
        - Parkinson's disease → use 'generate_mol_by_case' with arg 'Parkinson'.
        - Multiple sclerosis → use 'generate_mol_by_case' with arg 'Multiple sclerosis'. 
        - Dyslipidemia → use 'generate_mol_by_case' with arg 'Dyslipidemia'.
        - Acquired drug resistance → use 'generate_mol_by_case' with arg 'drug resistance'.
        - Lung cancer → use 'generate_mol_by_case' with arg 'Lung cancer'.

        IMPORTANT: Pre-defined disease models are GENERATIVE ONLY - they can generate new molecules
        but cannot predict properties. These models are always available without checking.

        2. CUSTOM USER-TRAINED MODELS (availability varies):
        - Use 'generate_mol_by_case' for generative custom models
        - Use 'predict_prop_by_smiles' for predictive custom models
        - First call 'get_state_from_server' (args: 'pred' or 'gen') to see ALL available models
        - This will show both pre-defined and custom models with their types (generative/predictive)
        - Custom model names are case-sensitive

        CRITICAL DIFFERENCES:
        - Pre-defined disease models: GENERATIVE ONLY, always available
        - Custom user-trained models: Can be GENERATIVE or PREDICTIVE, availability varies

        When a user requests molecule generation or property prediction:
        1. If it's for one of the pre-defined diseases above, use the specific generative tool for that disease
        2. For any other case, first call 'get_state_from_server' to check model availability and type. Call if the user wants to check the status of training.
        3. If a generative model exists, use 'generate_mol_by_case' with the exact model name
        4. If a predictive model exists, use 'predict_prop_by_smiles' with the exact model name
        5. If no suitable model exists, inform the user and suggest training it first
        For Dyslipidemia use 'gen_mols_dyslipidemia'. Etc.
        
        When a user requests to check status of training for specific case:
        1. Run 'get_case_state_from_server' with case name. For example: 'IC50_prediction'.

        Dataset for training from the user: {os.environ.get('DS_FROM_USER', False)} \n.
        Dataset for training from ChemBL (from llm agent): {os.environ.get('DS_FROM_CHEMBL', False)} \n
        Dataset for training from BindingDB (from llm agent): {os.environ.get('DS_FROM_BINDINGDB', False)} \n
        
        If you are asked about available predictive or generative models you should call get_state_from_server!!! And return list of case! 
        If you are asked to train a model, plan the training!
        ATTENTION: You are absolutely required to ALWAYS offer a tool for call! You are required to write all generated molecules in the final answer.

        If the path is written, it means that the user has uploaded their dataset, or the previous agent has transferred data. In this case, use the user's dataset, and if there is none, then the dataset from the agent (chose one of them) (be sure to use the full path). The user is a priority. If there is no path there, but you are asked to start training, you need to inform the user about this (write about this in your final answer).
        You must return the molecules without modifications. Do not lose symbols! All molecules must be transferred to the user.
        No more then 5 steps (tool calling)!!!
        
        So, your task from the user: """


memory_prompt = ChatPromptTemplate.from_template(
    """If the response suffers from the lack of memory, adjust it. Don't add any of your comments

Your objective is this:
input: {input};
response: {response};
memory {summary};
"""
)

worker_prompt = "You are a helpful assistant. You can use provided tools. \
    If there is no appropriate tool, or you can't use one, answer yourself"

automl_agent_description = """
'ml_dl_agent' - an agent that can run training of a generative model to generate SMILES, training of predictive models 
to predict properties. It also already stores ready-made models for inference. You can also ask him to prepare an 
existing dataset (you need to be specific in your request).
It can generate medicinal molecules. You must use this agent for molecules generation!!!\n
"""
dataset_builder_agent_description = "'dataset_builder_agent' - collects data from two databases - ChemBL and BindingDB. \
To collect data, it needs either the protein name or a specific id from a specific database. \
It can collect data from one specific database or from both. All data is saved locally. \
It also processes data: removes junk values, empty cells, and can filter if necessary.\n"
