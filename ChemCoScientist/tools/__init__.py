from langchain.tools.render import render_text_description

from ChemCoScientist.tools.chemist_tools import chem_tools, chem_tools_rendered
from ChemCoScientist.tools.ml_tools import (get_state_from_server,
                                            ml_dl_tools_rendered,
                                            predict_prop_by_smiles,
                                            train_ml_with_data)
from ChemCoScientist.tools.nano_tools import (nano_tools_rendered,
                                              nanoparticle_tools)
from ChemCoScientist.tools.web_tools import web_tools, web_tools_rendered

if web_tools:
    tools_rendered = render_text_description(
        web_tools + chem_tools + nanoparticle_tools
    )
else:
    tools_rendered = render_text_description(chem_tools + nanoparticle_tools)
