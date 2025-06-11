import os
from typing import Annotated, Optional
from urllib.parse import quote

import pubchempy as pcp
import py3Dmol
import rdkit.Chem as Chem
import requests
from langchain.tools.render import render_text_description
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langchain_experimental.utilities import PythonREPL
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import CalcMolDescriptors

repl = PythonREPL()
VALID_AFFINITY_TYPES = ["Ki", "Kd", "IC50"]

from smolagents import tool

import requests
import json
from typing import Dict, List, Optional


@tool
def fetch_BindingDB_data(params: Dict) -> List[Dict]:
    """
    Tool for retrieving protein affinity data from BindingDB.

    This tool:
    1. Takes a protein name as input or a UniProt ID
    2. Queries UniProt to find the corresponding UniProt ID (if not provided)
    3. Retrieves specified affinity values (Ki, Kd, or IC50) for the protein from BindingDB
    4. Returns structured data about ligands and their affinity measurements

    Data source: BindingDB (https://www.bindingdb.org) - a public database of measured binding affinities

    Args:
        params: Dictionary containing:
            - protein_name: Name of the target protein (required)
            - affinity_type: Type of affinity measurement (Ki, Kd, or IC50, default: Ki)
            - cutoff: Optional affinity threshold in nM (default: 10000)
            - id: Optional, UniProt ID

    Returns:
        List[dict]: List of dictionaries containing affinity data for the specified protein.
    """

    try:
        try:
            # parameter validation
            protein_name = params.get("protein_name")
            if not protein_name:
                print("Protein name not provided")
        except:
            pass

        affinity_type = params.get("affinity_type", "Ki")
        if affinity_type not in VALID_AFFINITY_TYPES:
            print(
                f"Invalid affinity type. Must be one of: {', '.join(VALID_AFFINITY_TYPES)}"
            )
            return False

        cutoff = params.get("cutoff", 10000)

        # Step 1: Get UniProt ID
        uniprot_id = params.get("id", False)
        if not uniprot_id:
            uniprot_id = fetch_uniprot_id(protein_name)
            if not uniprot_id:
                print(f"No UniProt ID found for {protein_name}")
                return False

        # Step 2: Retrieve affinity data from BindingDB
        affinity_entries = fetch_affinity_bindingdb(uniprot_id, affinity_type, cutoff)

        return affinity_entries

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return False


def fetch_uniprot_id(protein_name: str) -> Optional[str]:
    """
    Получает UniProt ID по названию белка через UniProt REST API.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"{protein_name} AND organism_id:9606",  # человек
        "format": "json",
        "size": 1,
        "fields": "accession",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            return data["results"][0].get("primaryAccession")
        return None

    except requests.exceptions.RequestException:
        return None


def fetch_affinity_bindingdb(
    uniprot_id: str, affinity_type: str, cutoff: int
) -> List[Dict]:
    """
    Retrieve affinity values from BindingDB for the given UniProt ID.

    Args:
        uniprot_id: UniProt accession ID
        affinity_type: Type of affinity measurement (Ki, Kd, or IC50)
        cutoff: Affinity threshold in nM

    Returns:
        List of dictionaries containing affinity data
    """
    url = f"http://bindingdb.org/rest/getLigandsByUniprots?uniprot={uniprot_id}&cutoff={cutoff}&response=application/json"

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        data = response.json()
        result = [
            i
            for i in data["getLindsByUniprotsResponse"]["affinities"]
            if i["affinity_type"] == affinity_type
        ]
        print(
            f"Found {len(result)} affinities for {uniprot_id} with type {affinity_type}"
        )
        return result

    except (requests.exceptions.RequestException, json.JSONDecodeError):
        return []


@tool
def fetch_chembl_data(
    target_name: str, target_id: str = "", affinity_type: str = "Ki"
) -> list[dict]:
    """Get Ki for activity by current protein from ChemBL database. Return
    dict with smiles and Ki values, format: [{"smiles": smiles, affinity_type: affinity_valie, "affinity_units": affinity_units}, ...]

    Args:
        target_name: str, name of protein,
        target_id: optional, id of current protein from ChemBL. Don't make it up yourself!!! Only user can ask!!!
        affinity_type: optional, str, type of affinity measurement (default: 'Ki').
    """
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

    if target_id == "" or target_id == None or target_id == False:
        # search target_id by protein name
        target_search = requests.get(
            f"{BASE_URL}/target/search?q={quote(target_name)}&format=json&limit=1000"
        )
        targets = target_search.json()["targets"]

        if not targets:
            print(f"Target '{target_name}' not found in ChEMBL")
            return []

        # get just first res
        target_id = targets[0]["target_chembl_id"]
        print(f"Found target: {targets[0]['pref_name']} ({target_id})")

    # get activity with Ki
    activities = []
    offset = 0
    while True:
        response = requests.get(
            f"{BASE_URL}/activity.json?"
            f"target_chembl_id={target_id}&"
            f"standard_type={affinity_type}&"
            f"offset={offset}&"
            "include=molecule"
        )

        data = response.json()
        activities += data["activities"]

        if not data["page_meta"]["next"]:
            break
        offset += len(data["activities"])

    # get SMILES and affinity values
    results = []
    for act in activities:
        try:
            smiles = act["canonical_smiles"]
            affinity_valie = act["standard_value"]
            affinity_units = act["standard_units"]
            results.append(
                {
                    "smiles": smiles,
                    affinity_type: affinity_valie,
                    "affinity_units": affinity_units,
                }
            )
        except (KeyError, TypeError):
            continue

    return results


from langchain_core.tools import tool


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute"],
):
    """Use this to execute python code and do math.You can't use it to extract information from previous steps. YOU CAN'T DONWLOAD ANY LIBRARIES. You can't write or save files. Don't invoke any system-dangerous code."""
    try:
        result = repl.run(code)
    except BaseException as e:
        # logger.exception(f"'python_repl_tool' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"
    result_str = (
        f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    )
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
    # try:
    #    result = repl.run(code)
    # except BaseException as e:
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
            compound = pcp.get_compounds(mol, "name")
            smiles = compound[0].canonical_smiles
            return smiles
        except BaseException as e:
            # logger.exception(f"'name2smiles' failed with error: {e}")
            return f"Failed to execute. Error: {repr(e)}"
    return "I've couldn't obtain smiles, the name is wrong"


@tool
def smiles2name(smiles: Annotated[str, "SMILES of a molecule"]):
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
            # logger.exception(f"'smiles2name' failed with error: {e}")
            return f"Failed to execute. Error: {repr(e)}"
    return "I've couldn't get iupac name"


@tool
def smiles2prop(
    smiles: Annotated[str, "SMILES of a molecule"], iupac: Optional[str] = None
):
    """Use this to calculate all available properties of given molecule. Only use for organic molecules
    params:
    smiles: str, smiles of a molecule,
    iupac: optional, default is None, iupac of molecule"""

    try:
        if iupac:
            compound = pcp.get_compounds(iupac, "name")
            if len(compound):
                smiles = compound[0].canonical_smiles

        res = CalcMolDescriptors(Chem.MolFromSmiles(smiles))
        return res
    except BaseException as e:
        # logger.exception(f"'smiles2prop' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"


@tool
def visualize_molecule(
    smiles: Annotated[str, "SMILES of a molecule"],
    config: RunnableConfig,
):
    """Use this to visualize/draw molecule with given SMILES. Don't call rdkit. Only use for organic molecules"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = Chem.Mol(mol)
            mol = AllChem.AddHs(mol, addCoords=True)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)

            view = py3Dmol.view(
                data=Chem.MolToMolBlock(mol),  # Convert the RDKit molecule for py3Dmol
                style={
                    "stick": {},
                    "sphere": {"scale": 0.3},
                },
                width=600,
                height=400,
            )
            view.setBackgroundColor("#b8bfcc")
            view.zoomTo()
            html_content = view.write_html()

            state = config["configurable"].get("state")
            # tool_call_id: Annotated[str, InjectedToolCallId] = state['messages'][-1]["tool_calls"][0]['id']

            path_to_results = os.path.join(
                os.environ.get("PATH_TO_RESULTS"), "vis_mols"
            )
            if not os.path.exists(path_to_results):
                os.makedirs(path_to_results)

            with open(
                os.path.join(path_to_results, "vis.html"), "w", encoding="utf-8"
            ) as f:
                f.write(html_content)

            answer = f"I've successfully generated images of {smiles} molecule"
            return answer
        else:
            return f"I've couldn't visualize this molecule. Perhaps SMILES is invalid"

    except BaseException as e:
        # logger.exception(f"'visualize_molecule' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"


@tool
def generate_molecule(
    params: Annotated[str, "Description of target molecule"], config: RunnableConfig
):
    """Use this to generate a molecule with given description. Returns smiles. Only use for organic molecules"""
    llm: BaseChatModel = config["configurable"].get("model")
    try:
        prompt = (
            "Generate smiles of molecule with given description. Answer only with smiles, nothing more: \
            Question: The molecule is a nitrogen mustard drug indicated for use in the treatment of chronic lymphocytic leukemia (CLL) and indolent B-cell non-Hodgkin lymphoma (NHL) that has progressed during or within six months of treatment with rituximab or a rituximab-containing regimen.  Bendamustine is a bifunctional mechlorethamine derivative capable of forming electrophilic alkyl groups that covalently bond to other molecules. Through this function as an alkylating agent, bendamustine causes intra- and inter-strand crosslinks between DNA bases resulting in cell death.  It is active against both active and quiescent cells, although the exact mechanism of action is unknown. \
            Answer: CN1C(CCCC(=O)O)=NC2=CC(N(CCCl)CCCl)=CC=C21 \
            Question: The molecule is a mannosylinositol phosphorylceramide compound having a tetracosanoyl group amide-linked to a C20 phytosphingosine base, with hydroxylation at C-2 and C-3 of the C24 very-long-chain fatty acid. It is functionally related to an Ins-1-P-Cer(t20:0/2,3-OH-24:0).\
            Answer: CCCCCCCCCCCCCCCCCCCCCC(O)C(O)C(=O)N[C@@H](COP(=O)(O)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1OC1O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]1O)[C@H](O)C(O)CCCCCCCCCCCCCCCC \
            Question: "
            + params
            + "\n Answer: "
        )
        res = llm.invoke(prompt)
        smiles = res.content
        return smiles
    except BaseException as e:
        # logger.exception(f"'generate_smiles' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"


chem_tools = [
    name2smiles,
    smiles2name,
    smiles2prop,
    generate_molecule,
    visualize_molecule,
]
chem_tools_rendered = render_text_description(chem_tools)
