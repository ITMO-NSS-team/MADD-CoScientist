from ChemCoScientist.dataset_handler.chembl.chembl_utils import ChemblLoader
import yaml
from openai import OpenAI
import json
from ChemCoScientist.agents.agents_prompts import ds_builder_prompt

with open("./ChemCoScientist/conf/conf.yaml", "r") as file:
    conf = yaml.safe_load(file)
    key = conf["api_key"]
    base_url = conf["base_url"]
    file_path = conf["chembl_csv_path"]


def dataset_builder_agent(state):
    pending_tasks = state["pending_tasks"]
    responses = state["responses"]

    if not pending_tasks:
        return {"done": "validate", "responses": responses}

    user_query = pending_tasks.pop(0)
    prompt = ds_builder_prompt + user_query + r"""You: """

    chembl_client = ChemblLoader(True, file_path)
    llm_client = OpenAI(api_key=key, base_url=base_url)

    response = llm_client.chat.completions.create(
        model="meta-llama/llama-3.1-70b-instruct",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
    )
    print("========")
    print(response.choices[0].message.content)
    print("========")

    try:
        query_params = eval(response.choices[0].message.content)
    except:
        try:
            query_params = eval(response.choices[0].message.content.split("You: ")[-1])
        except:
            return {
                "done": "error",
                "message": "Failed to parse LLM response.",
                "responses": None,
            }

    selected_columns = query_params.get("selected_columns", [])
    filters = query_params.get("filters", {})

    result_df = chembl_client.get_filtered_data(selected_columns, filters)
    responses.append(result_df)

    if not pending_tasks:
        return {"done": "validate", "responses": responses}
    else:
        return {"done": False, "pending_tasks": pending_tasks, "responses": responses}
