from typing import Union

import pandas as pd
import yaml
from agents import ConductorAgent, DecomposeAgent
from memory import ChatMemory
from MADD.diff_mas_versions.testcase.validate_pipeline import (compute_metrics,
                                                            validate_conductor,
                                                            validate_decompose)
from tools import (gen_mols_acquired_drug_resistance, gen_mols_all_case,
                   gen_mols_alzheimer, gen_mols_dyslipidemia,
                   gen_mols_lung_cancer, gen_mols_multiple_sclerosis,
                   gen_mols_parkinson, make_answer_chat_model,
                   request_mols_generation)

TOTAL_QUERYS = 0


class TestChain:
    """
    Chain for validation
    """

    def __init__(
        self,
        conductor_model: str,
        validation_path: str,
        api_vse_gpt: str = None,
        url: str = "",
        attempt: int = 1,
        is_many_funcs: bool = True,
        msg_for_store: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        conductor_model : str
            Name of model
        validate_model : str
            Name of model for validation by instruction
        api_vse_gpt : str, optional
            Api-key from personal account
        url: str
            Base url for OpenAI client
        valid_reflection : bool
            True if need validate by self reflection
        attempt : int, optional
            Number for attempts to fix function after faild validation
        is_many_funcs : bool, optional
            If True -> a separate generation function will be used for each case;
            if not -> one function will be used but with different arguments
        msg_for_store : int, optional
            Number of message for store
        """
        if len(api_vse_gpt) < 1:
            raise ValueError("API key for VSE GPT is missing.")
        self.attempt = attempt
        self.msg_for_store = msg_for_store
        self.chat_history = ChatMemory(
            msg_limit=msg_for_store, model_type=conductor_model
        )
        self.decompose_agent = DecomposeAgent(conductor_model)
        self.conductor_agent = ConductorAgent(
            conductor_model, api_key=api_vse_gpt, url=url, is_many_funcs=is_many_funcs
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

        self.validation_path = validation_path
        self.valid_tests, self.total_tests = 0, 0

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

    def task_handler(self, sub_task_number: int = 0) -> str:
        """Define tool for call and call it.
        Validate answer by self-reflection.

        Parameters
        ----------
        sub_task_number: int
            Number of subtask from decompose agent

        Returns
        -------
            Output from tools
        """
        global TOTAL_QUERYS

        tool = self.conductor_agent.call([self.chat_history.store[-1]])

        try:
            is_valid = validate_conductor(
                TOTAL_QUERYS, tool, sub_task_number, self.validation_path
            )
            if is_valid:
                self.valid_tests += 1
            self.total_tests += 1
        except Exception as e:
            self.total_tests += 1
            print("VALIDATION ERROR: ", e)

    def run(self, human_message: str, is_testing: bool) -> str:
        """Run chain"""
        global TOTAL_QUERYS

        tasks = self.decompose_agent.call(human_message)
        print("PROCESS: Define tasks: ", tasks)

        if is_testing:
            try:
                # validate decomposer agent
                validate_decompose(TOTAL_QUERYS, tasks, self.validation_path)
                # full result (after conductor)
                print(
                    "VALID_PASSED: ",
                    self.valid_tests,
                    "; VALID_PASSED_FROM:",
                    self.total_tests,
                )
            except:
                pass

        # limit for message store varies depending on the number of subtasks
        self.chat_history.msg_limit = self.chat_history.msg_limit * len(tasks)

        for i, task in enumerate(tasks):
            self.chat_history.add(task, "user")
            self.task_handler(i)
        TOTAL_QUERYS += 1


if __name__ == "__main__":
    with open("multi_agents_system/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    chain = TestChain(
        conductor_model=config["conductor_model"],
        api_vse_gpt=config["api_vse_gpt"],
        is_many_funcs=bool(config["is_many_funcs"]),
        attempt=int(config["attemps"]),
        validation_path=config["validation_path"],
        url=config["url"],
    )

    is_one_task_per_query = False
    questions = pd.read_excel(config["validation_path"]).values.tolist()

    for q in questions:
        response = chain.run(q[1], True)
        print(response)

    model_name = config["conductor_model"].replace("-", "_").replace("/", "_")
    compute_metrics(
        model_name=model_name,
        file_path=config["validation_path"],
        just_one_task_per_q=is_one_task_per_query,
    )
