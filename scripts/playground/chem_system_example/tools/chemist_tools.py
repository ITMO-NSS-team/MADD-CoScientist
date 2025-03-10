
from langchain_core.tools import tool
from langchain.tools.render import render_text_description
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_experimental.utilities import PythonREPL
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.types import Command


import pubchempy as pcp
from rdkit.Chem.Descriptors import CalcMolDescriptors
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import py3Dmol

import requests
from typing import Annotated, Optional
import os

import logging


# Create a separate logger for tools.py
logger = logging.getLogger("tools_logger")
logger.setLevel(logging.INFO)

# Configure a file handler for the tools logger
file_handler = logging.FileHandler("tools.log")
file_handler.setLevel(logging.INFO)

# Set a formatter for the tools logger
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the tools logger
logger.addHandler(file_handler)

# This executes code locally, which can be unsafe
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute"],
):
    """Use this to execute python code and do math.You can't use it to extract information from previous steps. YOU CAN'T DONWLOAD ANY LIBRARIES. You can't write or save files. Don't invoke any system-dangerous code."""
    try:
        result = repl.run(code)
    except BaseException as e:
        logger.exception(f"'python_repl_tool' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return result_str


@tool
def calc_prop_tool(
    smiles: Annotated[str, "The SMILES of a molecule"],
    property: Annotated[str, "The property to predict."],
):
    """Use this to predict molecular property.
    Can calculate refractive index and freezing point
    Do not call this tool more than once.
    Do not call another tool if this returns results."""
    #try:
    #    result = repl.run(code)
    #except BaseException as e:
    #    return f"Failed to execute. Error: {repr(e)}"
    result = 44.09
    result_str = f"Successfully calculated:\n\n{property}\n\nStdout: {result}"
    return result_str


@tool
def name2smiles(
    mol: Annotated[str, "Name of a molecule"],
):
    """Use this to convert molecule name to smiles format. Only use for organic molecules"""
    max_attempts = 3
    for attempts in range(max_attempts):
        try:
            compound = pcp.get_compounds(mol, 'name')
            smiles = compound[0].canonical_smiles
            return smiles
        except BaseException as e:
            logger.exception(f"'name2smiles' failed with error: {e}")
            return f"Failed to execute. Error: {repr(e)}"
    return "I've couldn't obtain smiles, the name is wrong"
    
@tool
def smiles2name(
    smiles: Annotated[str, "SMILES of a molecule"]
):
    """Use this to convert SMILES to IUPAC name of given molecule"""
    
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName/JSON"
    max_attempts = 3
    for attempts in range(max_attempts):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                iupac_name = data["PropertyTable"]["Properties"][0]["IUPACName"]
                return iupac_name 
            else:
                return "I've couldn't get iupac name"
            
        except BaseException as e:
            logger.exception(f"'smiles2name' failed with error: {e}")
            return f"Failed to execute. Error: {repr(e)}"
    return "I've couldn't get iupac name"
@tool
def smiles2prop(
    smiles: Annotated[str, "SMILES of a molecule"],
    iupac: Optional[str] = None
):
    """Use this to calculate all available properties of given molecule. Only use for organic molecules
    params:
    smiles: str, smiles of a molecule,
    iupac: optional, default is None, iupac of molecule"""
    
    try:
        if iupac:
            compound = pcp.get_compounds(iupac, 'name')
            if len(compound):
                smiles = compound[0].canonical_smiles

        res = CalcMolDescriptors(Chem.MolFromSmiles(smiles))
        return res
    except BaseException as e:
        logger.exception(f"'smiles2prop' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"
    
@tool
def visualize_molecule(smiles: Annotated[str, "SMILES of a molecule"], config: RunnableConfig,
):
    '''Use this to visualize/draw molecule with given SMILES. Don't call rdkit. Only use for organic molecules'''
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = Chem.Mol(mol)
            mol = AllChem.AddHs(mol, addCoords=True)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)

            view = py3Dmol.view(
                data=Chem.MolToMolBlock(mol),  # Convert the RDKit molecule for py3Dmol
                style={"stick": {}, "sphere": {"scale": 0.3},
                },
                width=600, height=400
    )
            view.setBackgroundColor('#b8bfcc')
            view.zoomTo()
            html_content = view.write_html()

            state = config['configurable'].get('state')
            #tool_call_id: Annotated[str, InjectedToolCallId] = state['messages'][-1]["tool_calls"][0]['id']

            path_to_results = os.path.join(os.environ.get('PATH_TO_RESULTS'), 'vis_mols')
            if not os.path.exists(path_to_results):
                os.makedirs(path_to_results)

            with open(os.path.join(path_to_results, 'vis.html'), "w", encoding="utf-8") as f:
                f.write(html_content)

            answer = f"I've successfully generated images of {smiles} molecule"
            return answer
        else:
            return f"I've couldn't visualize this molecule. Perhaps SMILES is invalid"
            '''
            return Command(
                update={
                    "visualization": html_content,
                    "messages": [
                        ToolMessage(
                            f"I've successfully visualized given molecule", tool_call_id=tool_call_id
                        )
                    ],
                }
            )
        else: 
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"I've couldn't visualize this molecule. Perhaps SMILES is invalid", tool_call_id=tool_call_id
                        )
                    ],
                }
            )'''
    except BaseException as e:
        logger.exception(f"'visualize_molecule' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"


@tool
def generate_molecule(
    params: Annotated[str, "Description of target molecule"],
    config: RunnableConfig
):
    """Use this to generate a molecule with given description. Returns smiles. Only use for organic molecules"""
    llm: BaseChatModel = config["configurable"].get("model")
    try:
        prompt = (
            'Generate smiles of molecule with given description. Answer only with smiles, nothing more: \
            Question: The molecule is a nitrogen mustard drug indicated for use in the treatment of chronic lymphocytic leukemia (CLL) and indolent B-cell non-Hodgkin lymphoma (NHL) that has progressed during or within six months of treatment with rituximab or a rituximab-containing regimen.  Bendamustine is a bifunctional mechlorethamine derivative capable of forming electrophilic alkyl groups that covalently bond to other molecules. Through this function as an alkylating agent, bendamustine causes intra- and inter-strand crosslinks between DNA bases resulting in cell death.  It is active against both active and quiescent cells, although the exact mechanism of action is unknown. \
            Answer: CN1C(CCCC(=O)O)=NC2=CC(N(CCCl)CCCl)=CC=C21 \
            Question: The molecule is a mannosylinositol phosphorylceramide compound having a tetracosanoyl group amide-linked to a C20 phytosphingosine base, with hydroxylation at C-2 and C-3 of the C24 very-long-chain fatty acid. It is functionally related to an Ins-1-P-Cer(t20:0/2,3-OH-24:0).\
            Answer: CCCCCCCCCCCCCCCCCCCCCC(O)C(O)C(=O)N[C@@H](COP(=O)(O)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1OC1O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]1O)[C@H](O)C(O)CCCCCCCCCCCCCCCC \
            Question: ' + params + '\n Answer: '
        )
        res = llm.invoke(prompt)
        smiles = res.content
        return smiles
    except BaseException as e:
        logger.exception(f"'generate_smiles' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"
    


chem_tools = [name2smiles, smiles2name, smiles2prop, generate_molecule, visualize_molecule]
chem_tools_rendered = render_text_description(chem_tools)
    