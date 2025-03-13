from ChemCoScientist.dataset_handler.chembl.chembl_utils import ChemblLoader

ds_builder_prompt = (
    r"""
    Extract relevant dataset filtering parameters from the following user request.
    Available columns:"""
    + str(ChemblLoader().get_columns())
    + r"""
    Return JSON in the format: {"selected_columns": [...], "filters": {{...}}}
    It is not necessary to specify filters (the dictionary with filters can be empty. In filters, you can specify ranges, string values, and Booleans.
    Required: Your answer must contain only language vocabulary (start answer from "{"). Use "float('-inf')" and "float('inf')" for negative and positive infinity (not None!).
        
    For example:
    User request: "Show molecules with molecular weight between 150 and 500."
    You: {'selected_columns': ['Molecular Weight"], "filters": {"Molecular Weight": (150, 500)}}
    Or:
    User request: "Show small molecules."
    You: {"selected_columns": ["Molecular Weight", "Type"], "filters": {"Type": "Small molecule"}}
        
    User request: 
    """
)

automl_prompt = """You are AutoML agent. 
You are obliged to call the tools (the most appropriate ones) for any user request and make your answer based on the results of the tools.

Rules:
1) Always call 'get_state_from_server' first to check if there is already a trained model with that name. 
If there is and the user wants to predict properties, run the prediction!
2) If you are asked to predict a property without a model name, you should get the state from the server (call 'get_state_from_server'), if it has a model that has a target with this property - call it!
"""
