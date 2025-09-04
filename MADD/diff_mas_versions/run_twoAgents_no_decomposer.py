import os
from os import listdir
from os.path import isfile, join
from typing import Union

import pandas as pd
import yaml
from additional_agents_for_exps import ConductorDecomposerAgent
from agents import ChatAgent, DecomposeAgent, SummaryAgent
from memory import ChatMemory
from prompting.props import enter, props_descp_dict, props_name
from testcase.validate_pipeline import (add_answers, check_total_answer,
                                        exctrac_mols_and_props,
                                        validate_conductor)
from MADD.diff_mas_versions.six_case_multi_agents_proto.tools import (gen_mols_acquired_drug_resistance, gen_mols_all_case,
                   gen_mols_alzheimer, gen_mols_dyslipidemia,
                   gen_mols_lung_cancer, gen_mols_multiple_sclerosis,
                   gen_mols_parkinson, make_answer_chat_model,
                   request_mols_generation)

TOTAL_QUERYS = 0


class ValidationChain:
    """
    Class for validation this version of MADD on dataset.
    """

    def __init__(
        self,
        conductor_model: str,
        llama_api_key: str = None,
        url: str = "",
        attempt: int = 1,
        is_many_funcs: bool = True,
        msg_for_store: int = 1,
        validation_path: str = "./experiment1.xlsx",
    ) -> None:
        """
        Parameters
        ----------
        conductor_model : str
            Name of model
        llama_api_key : str, optional
            Api-key from personal account
        url: str
            Base url for OpenAI client
        attempt : int, optional
            Number for attempts to fix function after faild validation
        is_many_funcs : bool, optional
            If True -> a separate generation function will be used for each case;
            if not -> one function will be used but with different arguments
        msg_for_store : int, optional
            Number of message for store
        """
        if len(llama_api_key) < 1:
            raise ValueError("API key for VSE GPT is missing.")
        self.attempt = attempt
        self.msg_for_store = msg_for_store
        self.chat_history = ChatMemory(
            msg_limit=msg_for_store, model_type=conductor_model
        )
        self.validation_path = validation_path
        self.decompose_agent = DecomposeAgent(conductor_model)
        self.conductor_agent = ConductorDecomposerAgent(
            model_name=conductor_model,
            api_key=llama_api_key,
            url=url,
            is_many_funcs=is_many_funcs,
        )
        self.chat_agent, self.summary_agent = (
            ChatAgent(model_name=conductor_model, api_key=llama_api_key, url=url),
            SummaryAgent(model_name=conductor_model, api_key=llama_api_key, url=url),
        )
        self.tools_map = {
            "request_mols_generation": request_mols_generation,
            "gen_mols_alzheimer": gen_mols_alzheimer,
            "gen_mols_multiple_sclerosis": gen_mols_multiple_sclerosis,
            "gen_mols_acquired_drug_resistance": gen_mols_acquired_drug_resistance,
            "gen_mols_dyslipidemia": gen_mols_dyslipidemia,
            "gen_mols_parkinson": gen_mols_parkinson,
            "gen_mols_lung_cancer": gen_mols_lung_cancer,
            "make_answer_chat_model": make_answer_chat_model,
            "gen_mols_all_case": gen_mols_all_case,
        }
        with open("/six_case_multi_agents_proto/config.yaml", "r") as file:
            self.conf = yaml.safe_load(file)

    def rm_last_saved_file(
        self, dir: str = "/six_case_multi_agents_proto/vizualization/"
    ):
        onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

        if onlyfiles != []:
            for file in onlyfiles:
                os.remove(dir + file)
            print("PROCESS: All files for vizualization deleted successfully")

    def call_tool(self, tool: dict) -> Union[list, int, str]:
        """
        Call tool like python function

        Parameters
        ----------
        tool : dict
            Name of function and parameters

        Returns
        answer : Union[list, int, str]
            Answer from current function
        """
        answer = self.tools_map[tool["name"].replace(" ", "")](tool["parameters"])
        return answer

    def task_handler(self, sub_task_number) -> str:
        """Define tool for call and call it.
        Validate answer by self-reflection.

        Returns
        -------
            Output from tools
        """
        global TOTAL_QUERYS
        total_attempt = 0
        tool = None

        # in case the format of the conductors answer is not correct
        try:
            if not (tool):
                tool = self.conductor_agent.call(self.chat_history.store)
                if isinstance(tool, list):
                    print("Success in format: tools in list.")
                    free_response_store, tables_store, mols = [], [], []
                    total_attempt += 1

                    for t in tool:
                        if total_attempt > self.attempt:
                            # if something went wrong - switch to next task
                            try:
                                is_valid = validate_conductor(
                                    TOTAL_QUERYS,
                                    "make_answer_chat_model",
                                    sub_task_number,
                                    self.validation_path,
                                )
                            except Exception as e:
                                print("VALIDATION ERROR: ", e)
                            sub_task_number += 1
                            free_response_store.append(
                                self.chat_agent.call(
                                    eval(self.chat_history.store[-1])["content"]
                                )
                            )
                            continue
                        res, mol = self.call_tool(t)
                        print("PROCESS: getted response from tool")

                        self.chat_history.add(str(tool), "assistant")
                        self.chat_history.add(str(res), "ipython")

                        if t == "make_answer_chat_model":
                            free_response_store.append(res)
                        else:
                            tables_store.append(res)
                            mols.append(mol)

                        try:
                            is_valid = validate_conductor(
                                TOTAL_QUERYS, t, sub_task_number, self.validation_path
                            )
                            sub_task_number += 1
                            print(is_valid)
                        except Exception as e:
                            sub_task_number += 1
                            print("VALIDATION ERROR: ", e)

                    return free_response_store, tables_store, mols
                else:
                    print("Warring: tools not in List.")
                    sub_task_number += 1
                    return False, False, False
        except:
            return False, False, False

    def run(self, human_message: str) -> str:
        """Run chain"""

        def collect_table_answer(finally_ans: str, tables: list) -> list:
            temp_prop = ""
            used = []
            for table in tables:
                for prop in props_name:
                    if prop in table and not (prop in used):
                        temp_prop += props_descp_dict[prop]
                        used.append(prop)
                finally_ans += table
                finally_ans += enter
            return [finally_ans, temp_prop]

        global TOTAL_QUERYS
        TOTAL_QUERYS += 1

        self.rm_last_saved_file()

        if human_message == "" or human_message == " ":
            response = "You didn't write anything - can I help you with something?"
            return response

        free_response_store, tables_store = [], []

        # limit for message store varies depending on the number of subtasks
        self.chat_history.msg_limit = 1 * len(human_message)
        self.chat_history.add(human_message, "user")
        free_response_store, tables_store, _ = self.task_handler(0)

        if isinstance(free_response_store, bool):
            return ["", "", True]

        finally_ans = ""

        # if there are more then 1 task
        if len(free_response_store) + len(tables_store) > 1:
            finally_ans, descp_props = collect_table_answer(finally_ans, tables_store)
            for free_ans in free_response_store:
                finally_ans += free_ans

            finally_ans = self.summary_agent.call(human_message, finally_ans)

        # if just 1 answer
        else:
            if free_response_store != []:
                finally_ans = free_response_store[0]
            else:
                finally_ans, descp_props = collect_table_answer(
                    finally_ans, tables_store
                )

        if tables_store != [] and descp_props != "":
            finally_ans += (
                enter + "Description of properties in table: \n" + descp_props
            )

        is_match_full = True
        true_mols, total_success = [], []

        if tables_store == []:
            return [finally_ans, tables_store, True]

        for j in range(len(tables_store)):
            true_mol = tables_store[j]
            mols_list = exctrac_mols_and_props(true_mol)
            true_mols.append(true_mol)
            if check_total_answer(mols_list, finally_ans) and is_match_full:
                continue
            else:
                is_match_full = False

        total_success.append(is_match_full)

        # for i in range(len(tables_store)):
        #     answers_store.append(finally_ans)

        return [finally_ans, tables_store, total_success[0]]


if __name__ == "__main__":
    answers_store, tables_st, total_succ = [], [], []
    with open("/six_case_multi_agents_proto/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    chain = ValidationChain(
        conductor_model=config["conductor_model"],
        llama_api_key=config["llama_api_key"],
        is_many_funcs=bool(config["is_many_funcs"]),
        attempt=int(config["attemps"]),
        url=config["url"],
        validation_path=config["validation_path"],
    )

    questions = pd.read_excel(config["validation_path"]).values.tolist()

    for i, q in enumerate(questions):
        answers, tables, success = chain.run(q[1])
        if isinstance(tables, str):
            answers_store.append(""), tables_st.append(""), total_succ.append(success)
        else:
            answers_store.append(answers), tables_st.append(
                tables[0]
            ), total_succ.append(success)
        add_answers(
            [answers_store, tables_st, total_succ], "./answers_no_decomposer_3.xlsx"
        )
