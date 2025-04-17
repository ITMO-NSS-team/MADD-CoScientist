import os
os.environ["OPENAI_API_KEY"] = "KEY"
os.environ["PATH_TO_DATA"] = "tools/models/datasets/image_dataset_multi_filtered"
os.environ["PATH_TO_CVAE_CHECKPOINT"] = "tools/models/checkpoints/cvae/model.pt"
os.environ["PATH_TO_RESULTS"] = "tools/generation_results"


from protollm.agents.builder import GraphBuilder
from protollm.connectors import create_llm_connector

from ChemCoScientist.agents.agents import (chemist_node, ml_dl_agent,
                                           nanoparticle_node, dataset_builder_agent)

from tools import (chem_tools_rendered, ml_dl_tools_rendered,
                   nano_tools_rendered, tools_rendered, dataset_handler_rendered)

model = create_llm_connector(
    "https://api.vsegpt.ru/v1;meta-llama/llama-3.1-70b-instruct"
)

visual_model = create_llm_connector(
    "https://api.vsegpt.ru/v1;vis-meta-llama/llama-3.2-90b-vision-instruct"
)

conf = {
    "recursion_limit": 50,
    "configurable": {
        "user_id": '1',
        "visual_model": visual_model,
        "img_path": "image.png",
        "llm": model,
        "max_retries": 1,
        "scenario_agents": [
            "chemist_node",
            "nanoparticle_node",
            "ml_dl_agent",
        ],
        "scenario_agent_funcs": {
            "chemist_node": chemist_node,
            "nanoparticle_node": nanoparticle_node,
            "ml_dl_agent": ml_dl_agent,
            "dataset_builder_agent": dataset_builder_agent,
        },
        "tools_for_agents": {
            # here can be description of langchain web tools (not TavilySearch)
            # "web_serach": [web_tools_rendered],
            "chemist_node": [chem_tools_rendered],
            "nanoparticle_node": [nano_tools_rendered],
            "ml_dl_agent": [ml_dl_tools_rendered],
            "dataset_builder_agent": [dataset_handler_rendered]
        },
        # here can be langchain web tools (not TavilySearch)
        # "web_tools": web_tools,
        "tools_descp": tools_rendered,
        "web_search": True
    },
}

# UNCOMMENT the one you need

# inputs = {"input": "Посчитай qed, tpsa, logp, hbd, hba свойства ацетона"}
# inputs = {"input": "Поищи с помощью поиска свежие статьи на тему онлайн-синтеза наноматериалов"}
# inputs = {"input": "Generate an image for nanoparticles most optimal for catalysis."}
# inputs = {"input": "Что ты можешь?"}
# inputs = {"input": "Узнай есть ли обученные модели, напиши какие именно доступны."}
# inputs = {"input": "Запусти предсказание с помощью мл-модели на значение IC50 для молекулы Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1."}
# inputs = {"input": "Запусти обучение на данных из /Users/alina/Desktop/ИТМО/ChemCoScientist/ChemCoScientist/dataset_handler/chembl/docking.csv. В качестве таргета возьми Docking score. Обязательно назови кейс DOCKING_SCORE"}
# inputs = {"input": "Предскажи Docking score для Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1 с помощью мл-модели."}
# inputs = {"input": "Модель с названием DOCKING_SCORE еще обучается?"}
# inputs = {"input": "Обучи модель на данных из ChemBl предсказывать значение IC50. Модель сохрани с названием 'chembl_ic50'."}
inputs = {"input": "Найди информацию о последних открытиях в области лечения Рака Легкого на последней стадии. Дай ссылку на контент."}

if __name__ == "__main__":
    graph = GraphBuilder(conf)
    res_1 = graph.run(inputs, debug=True, user_id="1")
