import os

from langchain_core.prompts import ChatPromptTemplate

ds_builder_prompt = f"You can download data from ChemBL, BindingDB. \n\
Rules: \n\
1) Don't call downloading from ChemBL, BindingDB unless they ask you to download! \n\
2) Never invent IDs from the database yourself. Specify them only if the user names them himself.\n\
3) Don't change the protein name from the user's request. If they ask for SARS-CoV-2, then pass the protein_name unchanged.\n\
"

automl_prompt = f"""You have access to two types of generative models:

        1. PRE-DEFINED DISEASE MODELS (always available):
        - Alzheimer's disease → use 'generate_mol_by_case' with arg 'Alzheimer'. Description of disease: GSK-3beta inhibitors with high activity. \
    These molecules can bind to GSK-3beta protein, molecules has low brain-blood barrier permeability
        
        - Parkinson's disease → use 'generate_mol_by_case' with arg 'Parkinson'.
        
        - Multiple sclerosis → use 'generate_mol_by_case' with arg 'Multiple sclerosis'. Description of disease: There are high activity tyrosine-protein kinase BTK inhibitors or highly potent non-covalent \
    BTK tyrosine kinase inhibitors from the TEC family of tyrosine kinases that have the potential \
    to affect B cells as a therapeutic target for the treatment of multiple sclerosis.
        
        - Dyslipidemia → use 'generate_mol_by_case' with arg 'Dyslipidemia'. Description of disease:     Generation of molecules for the treatment of dyslipidemia.\
    Molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and \
    the ability to cross the BBB. Molecules have affinity to the protein ATP citrate synthase, enhances reverse cholesterol transport via ABCA1 upregulation\
    , inhibits HMG-CoA reductase with improved safety profile compared to statins. It can be  PCSK9 inhibitors to enhance LDL receptor recycling and reduce LDL cholesterol levels.",
        
        - Acquired drug resistance → use 'generate_mol_by_case' with arg 'drug resistance'. Description of disease: Generation of molecules for acquired drug resistance. \
    Molecules that selectively induce apoptosis in drug-resistant tumor cells.\
    It significantly enhances the activity of existing therapeutic agents against drug-resistant pathogens.

        - Lung cancer → use 'generate_mol_by_case' with arg 'Lung cancer'.Description of disease: Molecules are inhibitors of KRAS protein with G12C mutation. \
    The molecules are selective, meaning they should not bind with HRAS and NRAS proteins.\
    Its target KRAS proteins with all possible mutations, including G12A/C/D/F/V/S, G13C/D, \
    V14I, L19F, Q22K, D33E, Q61H, K117N and A146V/T.
    
        Attention! You should be able to understand from the request, without the case name, what molecule needs to be generated!
        For example, the question: 'Generate GSK-3beta inhibitors with high docking score' - refers to Alzheimer's.
        Example: 'Suggest several molecules that have high docking affinity with KRAS G12C protein. Molecules should possess common drug-like properties, including low toxicity, high QED score, and high level of synthesizability.' to lung cancer.
        Example: "Develop kinase-binding agents that specifically inhibit Bruton's tyrosine kinase for therapeutic use in multiple sclerosis." to sclerosis.

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
        2. For any other case, first call 'get_state_from_server' to check model availability and type
        3. If a generative model exists, use 'generate_mol_by_case' with the exact model name
        4. If a predictive model exists, use 'predict_prop_by_smiles' with the exact model name
        5. If no suitable model exists, inform the user and suggest training it first
        For Dyslipidemia use 'gen_mols_dyslipidemia'. Etc.

        Dataset for training from the user: {os.environ.get('DS_FROM_USER', False)} \n.
        Dataset for training from ChemBL (from llm agent): {os.environ.get('DS_FROM_CHEMBL', False)} \n
        Dataset for training from BindingDB (from llm agent): {os.environ.get('DS_FROM_BINDINGDB', False)} \n
        
        If you are asked about available predictive or generative models you should call get_state_from server!!! And return list of case! 
        If you are asked to train a model, plan the training!

        If the path is written, it means that the user has uploaded their dataset, or the previous agent has transferred data. In this case, use the user's dataset, and if there is none, then the dataset from the agent (chose one of them) (be sure to use the full path). The user is a priority. If there is no path there, but you are asked to start training, you need to inform the user about this (write about this in your final answer).
        You must return the molecules without modifications. Do not lose symbols! All molecules must be transferred to the user.
        No more then 3 steps (tool calling)!!!
        
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
