import os

from dotenv import load_dotenv
load_dotenv("config.env")

from protollm.agents.builder import GraphBuilder
from protollm.connectors import create_llm_connector

from MADD.mas.prompts.prompts import (
    automl_agent_description,
    dataset_builder_agent_description,
    dataset_processing_agent_description,
)
from MADD.mas.scenarion_agents import dataset_builder_agent, ml_dl_agent, dataset_processing_agent


def create_by_default_setup() -> GraphBuilder:
    functional_description = (
        automl_agent_description + dataset_builder_agent_description + dataset_processing_agent_description
    )
    conf = {
        # maximum number of recursions (be careful!)
        "recursion_limit": 50,
        "configurable": {
            "user_id": "1",
            "visual_model": create_llm_connector(f"{os.environ['VISION_LLM_URL']}", temperature=0.0),
            "img_path": "image.png",
            "llm": create_llm_connector(
                f"{os.environ['MAIN_LLM_URL']};{os.environ['MAIN_LLM_MODEL']}", temperature=0.0
            ),
            "max_retries": 3,
            # list of scenario agents
            "scenario_agents": [
                "ml_dl_agent",
                "dataset_builder_agent",
                "dataset_processing_agent",
            ],
            # nodes for scenario agents
            "scenario_agent_funcs": {
                "ml_dl_agent": ml_dl_agent,
                "dataset_builder_agent": dataset_builder_agent,
                 "dataset_processing_agent": dataset_processing_agent,
            },
            # descripton for agents tools - if using langchain @tool
            # or description of agent capabilities in free format
            "tools_for_agents": {
                "dataset_builder_agent": [dataset_builder_agent_description],
                "ml_dl_agent": [automl_agent_description],
                "dataset_processing_agent": [dataset_processing_agent_description],
            },
            # full descripton for agents tools
            "tools_descp": functional_description,
            # add a key with the agent node name if you need to pass something to it
            "additional_agents_info": {
                "dataset_builder_agent": {
                    "model_name": os.environ["SCENARIO_LLM_MODEL"],
                    "url": os.environ["SCENARIO_LLM_URL"],
                    "api_key": os.environ["OPENAI_API_KEY"],
                    "ds_dir": os.environ["DS_STORAGE_PATH"],
                },
                "ml_dl_agent": {
                    "model_name": os.environ["SCENARIO_LLM_MODEL"],
                    "url": os.environ["SCENARIO_LLM_URL"],
                    "api_key": os.environ["OPENAI_API_KEY"],
                    "ds_dir": os.environ["DS_STORAGE_PATH"],
                },
                "dataset_processing_agent": {
                "model_name": os.environ["SCENARIO_LLM_MODEL"],
                "url": os.environ["SCENARIO_LLM_URL"],
                "api_key": os.environ["OPENAI_API_KEY"],
                "ds_dir": os.environ["DS_STORAGE_PATH"],
                },
            },
            # These prompts will be added in ProtoLLM
            "prompts": {
                "supervisor": {
                    "problem_statement": None,
                    "problem_statement_continue": None,
                    "rules": None,
                    "additional_rules": None,
                    "examples": None,
                    "enhancemen_significance": None,
                },
                "planner": {
                    "problem_statement": "You are an AI planning agent. Your sole task is to analyze the user's input query and generate a concise, efficient, and minimal plan to fulfill it.",
                    "rules": """Rule for Generation/Prediction:

                If the user's question is exclusively about generating molecules (e.g., "generate novel molecules similar to Aspirin") or predicting properties (e.g., "predict the solubility of this molecule"),
                Then the plan must consist of a single step: Instruct the ml_dl_agent to handle the request.

                Your response must be exactly this:
                1. Generate the molecules/properties using the ml_dl_agent.

                Rule for Model Training:

                If the request involves preparing a dataset and training a model (e.g., "train a model to predict toxicity", "create a QSAR model for this data"),

                Then the plan must follow this strict two-step sequence:
                1. Download the required dataset.
                2. Initiate training on the downloaded data.
                
                If you are asked to generate something, try to immediately understand which case it relates to and indicate the name of the disease in the plan.
                'Alzheimer' - generation of drug molecules for the treatment of Alzheimer's disease. \
                GSK-3beta inhibitors with high activity. \
                These molecules can bind to GSK-3beta protein, molecules has low brain-blood barrier permeability.\
                Compounds that contain heterocyclic moieties to enhance binding affinity for amyloid-beta aggregates.\
                Hot keys: inhibitory activity against glycogen synthase kinase 3 beta (GSK-3β); compound should \
                demonstrate a permeability coefficient of at least 1.5 to ensure effective crossing of the blood-brain barrier; \
                tau protein kinases with an IC50 value lower than 50 nM.\
                'Sclerosis' - Generation of molecules for the treatment of multiple sclerosis.\
                There are high activity tyrosine-protein kinase BTK inhibitors or highly potent non-covalent \
                BTK tyrosine kinase inhibitors from the TEC family of tyrosine kinases that have the potential \
                to affect B cells as a therapeutic target for the treatment of multiple sclerosis.\
                Hot keys: high activity tyrosine-protein kinase BTK inhibitors;  inhibit Bruton's tyrosine kinase \
                through non-covalent interaction; non-covalent BTK inhibitors with enhanced permeability across the blood-brain \
                barrier and high selectivity for Cytoplasmic tyrosine-protein kinase BMX;  immune signaling pathways \
                to treat multiple sclerosis.\
                'Parkinson' - Generation of molecules for the treatment of Parkinson's disease.\
                These compounds should possess high bioavailability, cross the blood-brain barrier efficiently, and show \
                minimal metabolic degradation.\
                Hot keys: derivatives from amino acid groups with modifications in side chains to enhance bioactivity; \
                heterocyclic compounds featuring nitrogen and sulfur to improve pharmacokinetics; molecules using a \
                fragment-based approach, combining elements of natural alkaloids; molecules with properties of glutamate \
                receptor antagonists for neuroprotection; compounds that inhibit monoamine oxidase B (MAO-B);\
                dopamine agonist with improved blood-brain barrier penetration; dual-action molecule combining D2 \
                receptor agonism and A2A antagonism;  PDE10A inhibitor with enhanced selectivity and potency.\
                'Lung Cancer' - Generation of molecules for the treatment of lung cancer. \
                Molecules are inhibitors of KRAS protein with G12C mutation. \
                The molecules are selective, meaning they should not bind with HRAS and NRAS proteins.\
                Its target KRAS proteins with all possible mutations, including G12A/C/D/F/V/S, G13C/D, \
                V14I, L19F, Q22K, D33E, Q61H, K117N, G12C and A146V/T.\
                Hot keys: HRAS and NRAS proteins; KRAS G12C protein mutation, which drives cancer growth in lung cancer;\
                avoiding binding to HRAS and NRAS; low cross-reactivity with other RAS isoforms;  molecules to specifically bind and inhibit KRAS G12C\
                'dyslipidemia' - Generation of molecules for the treatment of dyslipidemia.\
                Molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and \
                the ability to cross the BBB. Molecules have affinity to the protein ATP citrate synthase, enhances reverse cholesterol transport via ABCA1 upregulation\
                , inhibits HMG-CoA reductase with improved safety profile compared to statins. It can be  PCSK9 inhibitors to enhance LDL receptor recycling and reduce LDL cholesterol levels.\
                Hot keys: molecules that disrupt the interaction between CD36 and oxidized LDL; ligands for SREBP-1c \
                inhibition to regulate fatty acid and triglyceride synthesis; dual-action agents that modulate both HDL \
                and LDL for improved cardiovascular outcomes; AMPK pathway to promote fatty acid oxidation;\
                IC50 value lower than 50 µM for inhibiting CETP;  Tmax of less than 2 hours for rapid action in lipid regulation;\
                a negative logD at pH 7.4 for improved selectivity in tissues; ANGPTL3 inhibitor to reduce plasma triglycerides;\
                PPARα agonist with reduced side;  ApoC-III antisense oligonucleotide with enhanced cellular uptake.\
                'drug resistance' - Generation of molecules for acquired drug resistance. \
                Molecules that selectively induce apoptosis in drug-resistant tumor cells.\
                It significantly enhances the activity of existing therapeutic agents against drug-resistant pathogens.\
                Hot keys: molecular structures targeting drug resistance mechanisms in cancer cells;\
                ABC transporters involved in multidrug resistance; molecules that selectively induce apoptosis in drug-resistant tumor cells;\
                counteracting drug resistance in oncological therapies; treatment sensitivity in resistant tumors; compounds that enhance the \
                efficacy of existing anti-resistance treatments; synergistic compound that significantly enhances the activity of existing therapeutic \
                agents against drug-resistant pathogens; selectively target the Ras-Raf-MEK-ERK signaling pathway\
                molecules targeting the efflux pumps responsible for drug resistance.\
                """,
                    "desc_restrictions": None,
                    "examples": """Request: "Design molecules that specifically inhibit KRAS G12C, a target for lung cancer treatment. These molecules should have high binding affinity for KRAS G12C and low cross-reactivity with other RAS isoforms"
                            Response: {
                                "steps": [
                                    ["Generate molecule for case 'Lung Cancer'"]
                                ]
                            }
                            
                            Example:
                            Request: "Train model to predict IC50 on my data."
                            Response: {
                                "steps": 
                                    [['Train model to predict IC50 on users data. Select IC50 column, as target.']]
                            }
                            
                            Example:
                            Request: ""Generate GSK-3beta inhibitors with high activit. Suggest some small molecules that inhibit KRAS G12C - a target responsible for non-small cell lung cancer. Generate high activity tyrosine-protein kinase BTK inhibitors. Can you suggest molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and the ability to cross the BBB?"
                            Response: {
                                "steps": 
                                    [['Generate 1 molecule by model with name "Alzheimer"'], ['Generate 1 molecule by model "Lung Cancer"'], ['Generate 1 molecule by model "sclerosis"'], ['Generate 1 molecule by model "dyslipidemia"']]
                            }
                            """,
                    "additional_hints": "If the user provides his dataset - \
                        immediately start training using ml_dl_agent (never call dataset_builder_agent)!",
                },
                "chat": {
                    "problem_statement": None,
                    "additional_hints": """You are a chemical agent system. You can do the following:
                        - train generative models (generate SMILES molecules), train predictive models (predict properties)
                        - prepare a dataset for training
                        - download data from chemical databases: ChemBL, BindingDB
                        - generate molecules for this cases:
                            1) for the treatment of Alzheimer's disease. 
                            2) for the treatment of Parkinson's disease. 
                            3) for the treatment of multiple sclerosis.  
                            4) for the treatment of dyslipidemia.  
                            5) for acquired drug resistance. 
                            6) for the treatment of lung cancer. 
                            (Specify the name of the disease and the number of molecules to generate)
                                                    
                        If user ask something like "What can you do" - make answer yourself!
                        If you are asked to generate something, you are not allowed to come up with something yourself, pass the question on.
                        """,
                },
                "summary": {
                    "problem_statement": None,
                    "rules": None,
                    "additional_hints": "Never write full paths! Only file names. You must output all the molecules (if they exist)! And all the tables (if exist)! You must return the molecules without modifications.Make a beautiful answer, if something is missing, don’t write about it.\
                        Always add these cases to your list of available models:\
                            Alzheimer, Parkinson, Sclerosis, Lung Cancer, dyslipidemia,drug resistance\
",
                },
                "replanner": {
                    "problem_statement": None,
                    "rules": None,
                    "examples": None,
                    "additional_hints": """Attention: you must delete completed steps from the plan! Pay close attention to past_steps, do not plan what is done there. This is very, very important.\
                        
                    Don't plan to predict anything if the molecules are generated and have some properties! Return the answer. 
                    BUT if there are no molecules for all cases in the plan - replan them! Adapt the plan.
                    
                    If you are asked to generate something, try to immediately understand which case it relates to and indicate the name of the disease in the plan.
                    'Alzheimer' - generation of drug molecules for the treatment of Alzheimer's disease. \
                    GSK-3beta inhibitors with high activity. \
                    These molecules can bind to GSK-3beta protein, molecules has low brain-blood barrier permeability.\
                    Compounds that contain heterocyclic moieties to enhance binding affinity for amyloid-beta aggregates.\
                    Hot keys: inhibitory activity against glycogen synthase kinase 3 beta (GSK-3β); compound should \
                    demonstrate a permeability coefficient of at least 1.5 to ensure effective crossing of the blood-brain barrier; \
                    tau protein kinases with an IC50 value lower than 50 nM.\
                    'Sclerosis' - Generation of molecules for the treatment of multiple sclerosis.\
                    There are high activity tyrosine-protein kinase BTK inhibitors or highly potent non-covalent \
                    BTK tyrosine kinase inhibitors from the TEC family of tyrosine kinases that have the potential \
                    to affect B cells as a therapeutic target for the treatment of multiple sclerosis.\
                    Hot keys: high activity tyrosine-protein kinase BTK inhibitors;  inhibit Bruton's tyrosine kinase \
                    through non-covalent interaction; non-covalent BTK inhibitors with enhanced permeability across the blood-brain \
                    barrier and high selectivity for Cytoplasmic tyrosine-protein kinase BMX;  immune signaling pathways \
                    to treat multiple sclerosis.\
                    'Parkinson' - Generation of molecules for the treatment of Parkinson's disease.\
                    These compounds should possess high bioavailability, cross the blood-brain barrier efficiently, and show \
                    minimal metabolic degradation.\
                    Hot keys: derivatives from amino acid groups with modifications in side chains to enhance bioactivity; \
                    heterocyclic compounds featuring nitrogen and sulfur to improve pharmacokinetics; molecules using a \
                    fragment-based approach, combining elements of natural alkaloids; molecules with properties of glutamate \
                    receptor antagonists for neuroprotection; compounds that inhibit monoamine oxidase B (MAO-B);\
                    dopamine agonist with improved blood-brain barrier penetration; dual-action molecule combining D2 \
                    receptor agonism and A2A antagonism;  PDE10A inhibitor with enhanced selectivity and potency.\
                    'Lung Cancer' - Generation of molecules for the treatment of lung cancer. \
                    Molecules are inhibitors of KRAS protein with G12C mutation. \
                    The molecules are selective, meaning they should not bind with HRAS and NRAS proteins.\
                    Its target KRAS proteins with all possible mutations, including G12A/C/D/F/V/S, G13C/D, \
                    V14I, L19F, Q22K, D33E, Q61H, K117N, G12C and A146V/T.\
                    Hot keys: HRAS and NRAS proteins; KRAS G12C protein mutation, which drives cancer growth in lung cancer;\
                    avoiding binding to HRAS and NRAS; low cross-reactivity with other RAS isoforms;  molecules to specifically bind and inhibit KRAS G12C\
                    'dyslipidemia' - Generation of molecules for the treatment of dyslipidemia.\
                    Molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and \
                    the ability to cross the BBB. Molecules have affinity to the protein ATP citrate synthase, enhances reverse cholesterol transport via ABCA1 upregulation\
                    , inhibits HMG-CoA reductase with improved safety profile compared to statins. It can be  PCSK9 inhibitors to enhance LDL receptor recycling and reduce LDL cholesterol levels.\
                    Hot keys: molecules that disrupt the interaction between CD36 and oxidized LDL; ligands for SREBP-1c \
                    inhibition to regulate fatty acid and triglyceride synthesis; dual-action agents that modulate both HDL \
                    and LDL for improved cardiovascular outcomes; AMPK pathway to promote fatty acid oxidation;\
                    IC50 value lower than 50 µM for inhibiting CETP;  Tmax of less than 2 hours for rapid action in lipid regulation;\
                    a negative logD at pH 7.4 for improved selectivity in tissues; ANGPTL3 inhibitor to reduce plasma triglycerides;\
                    PPARα agonist with reduced side;  ApoC-III antisense oligonucleotide with enhanced cellular uptake.\
                    'drug resistance' - Generation of molecules for acquired drug resistance. \
                    Molecules that selectively induce apoptosis in drug-resistant tumor cells.\
                    It significantly enhances the activity of existing therapeutic agents against drug-resistant pathogens.\
                    Hot keys: molecular structures targeting drug resistance mechanisms in cancer cells;\
                    ABC transporters involved in multidrug resistance; molecules that selectively induce apoptosis in drug-resistant tumor cells;\
                    counteracting drug resistance in oncological therapies; treatment sensitivity in resistant tumors; compounds that enhance the \
                    efficacy of existing anti-resistance treatments; synergistic compound that significantly enhances the activity of existing therapeutic \
                    agents against drug-resistant pathogens; selectively target the Ras-Raf-MEK-ERK signaling pathway\
                    molecules targeting the efflux pumps responsible for drug resistance.
                    
                    Example:
                    Request: Generate GSK-3beta inhibitors with high activit. Suggest some small molecules that inhibit KRAS G12C - a target responsible for non-small cell lung cancer. Generate high activity tyrosine-protein kinase BTK inhibitors. Can you suggest molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and the ability to cross the BBB?"
                    Response: {
                        "steps": [
                            [['Generate 1 molecule by model with name "Alzheimer"'], ['Generate 1 molecule by model "Lung Cancer"'], ['Generate 1 molecule by model "sclerosis"'], ['Generate 1 molecule by model "dyslipidemia"']]
                        ]
                    }
                    Example:
                        Request: "Train model to predict IC50 on my data."
                        Response: {
                            "steps": 
                                [['Train model to predict IC50 on users data. Select IC50 column, as target.']]
                    }
                    
                    If you are asked to train a model, plan the training! 
                    \
                    Optimize the plan, transfer already existing answers from previous executions.\
                    Don't forget tasks.\You must return the molecules without modifications. Do not lose symbols. All molecules must be transferred to the user.\n\
                    Be more careful about which tasks can be performed in parallel and which ones can be performed sequentially.\
                    For example, you cannot fill a table and save it in parallel.""",
                },
            },
        },
    }
    return GraphBuilder(conf)
