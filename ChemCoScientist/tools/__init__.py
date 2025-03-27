from ChemCoScientist.tools.web_tools import web_tools, web_tools_rendered
from ChemCoScientist.tools.nano_tools import nanoparticle_tools, nano_tools_rendered
from ChemCoScientist.tools.chemist_tools import chem_tools, chem_tools_rendered
from ChemCoScientist.tools.ml_tools import predict_prop_by_smiles, train_ml_with_data, get_state_from_server, ml_dl_tools_rendered


from langchain.tools.render import render_text_description
if web_tools:
    tools_rendered = render_text_description(web_tools+chem_tools+nanoparticle_tools)
else: 
    tools_rendered = render_text_description(chem_tools+nanoparticle_tools)