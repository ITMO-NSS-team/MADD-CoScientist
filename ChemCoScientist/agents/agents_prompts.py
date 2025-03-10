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
