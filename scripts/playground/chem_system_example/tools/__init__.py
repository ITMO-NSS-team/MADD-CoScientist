from tools.web_tools import web_tools, web_tools_rendered
from tools.nano_tools import nanoparticle_tools, nano_tools_rendered
from tools.chemist_tools import chem_tools, chem_tools_rendered


from langchain.tools.render import render_text_description
if web_tools:
    tools_rendered = render_text_description(web_tools+chem_tools+nanoparticle_tools)
else: 
    tools_rendered = render_text_description(chem_tools+nanoparticle_tools)