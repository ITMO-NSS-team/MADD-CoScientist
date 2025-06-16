import datetime
import logging
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
import pandas as pd
from protollm.metrics import model_for_metrics, correctness_metric

from ChemCoScientist.answer_question import query_llm
from chromadb.chroma_db_operations import query_chromadb

path_to_data = ""
all_questions = pd.read_csv(Path(path_to_data, "dataset.csv"))

metrics_init_params = {
    "model": model_for_metrics,
    "verbose_mode": True,
    "async_mode": False,
}
answer_relevancy = AnswerRelevancyMetric(**metrics_init_params)
faithfulness = FaithfulnessMetric(**metrics_init_params)
context_precision = ContextualPrecisionMetric(**metrics_init_params)
context_recall = ContextualRecallMetric(**metrics_init_params)
context_relevancy = ContextualRelevancyMetric(**metrics_init_params)

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def retrieve_context(collection, question: str) -> (str, list):
    txt_context = ''
    img_paths = []
    
    context = query_chromadb(collection, question)
    
    for ind, chunk in enumerate(context['metadatas'][0]):
        if chunk['type'] == 'image':
            img_paths.append(chunk['image_path'])
        if chunk['type'] == 'text':
            txt_context += '\n\n' + context['documents'][0][ind]
    return txt_context, img_paths


class Timer:
    def __init__(self):
        self.process_terminated = False

    def __enter__(self):
        self.start = datetime.datetime.now()
        return self

    @property
    def start_time(self):
        return self.start

    @property
    def spent_time(self) -> datetime.timedelta:
        return datetime.datetime.now() - self.start_time

    @property
    def seconds_from_start(self) -> float:
        return round(self.spent_time.total_seconds(), 2)

    def __exit__(self, *args):
        return self.process_terminated


def pipeline_test_with_save(
        data: pd.DataFrame, metrics_to_calculate: list, model_name: str, model_url: str, version: float
) -> pd.DataFrame:
    """Tests pipeline.

    Args:
        data: questions, correct context/answer etc.
        metrics_to_calculate: list of metrics to be calculated
        model_name: string with model name
        model_url: string with model URL and name
        version: test version

    Returns: pandas DataFrame
    """
    _log.info("Pipeline test is running...")
    out_dir = Path("./test_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    path_to_results = Path(out_dir, f"pipeline_test_{model_name}_v{version}.txt")
    path_to_df = Path(out_dir, f"pipeline_test_{model_name}_v{version}.csv")
    
    columns = [
        "index", "question", "correct_context", "context_from_db",
        "correct_answer", "answer_from_model", "context_retrieve_time",
        "answer_generation_time"
    ]
    for metric in metrics_to_calculate:
        columns.append(f"{metric.__name__}_score")
        columns.append(f"{metric.__name__}_reason")
    
    if path_to_df.exists():
        existing_df = pd.read_csv(path_to_df)
        clear_existing_df = existing_df.drop_duplicates(subset=["index"], keep=False)
        clear_existing_df.to_csv(path_to_df, index=False)
        processed_indices = clear_existing_df["index"].unique().tolist() if "index" in clear_existing_df.columns else []
        start_index = max(processed_indices) + 1 if processed_indices else 0
    else:
        existing_df = pd.DataFrame(columns=columns)
        existing_df.to_csv(path_to_df, index=False)
        start_index = 0
    
    for i, row in data.iterrows():
        if i < start_index:
            continue
        
        try:
            _log.info(f"Processing question {i}")
            question = row["question"].replace('"', "'")
            correct_answer = row["correct_answer"]
            correct_context = row["correct_context"]
            col = row["collection_name"]
            
            row_data = {
                "index": i,
                "question": question,
                "correct_context": correct_context,
                "context_from_db": "",
                "correct_answer": correct_answer,
                "answer_from_model": "",
                "context_retrieve_time": None,
                "answer_generation_time": None
            }
            
            for metric in metrics_to_calculate:
                row_data[f"{metric.__name__}_score"] = -1
                row_data[f"{metric.__name__}_reason"] = ""
            
            with Timer() as t:
                try:
                    context, imgs = retrieve_context(col, question)
                    row_data["context_retrieve_time"] = t.seconds_from_start
                except Exception as e:
                    _log.error(f"Context retrieval failed: {str(e)}")
                    context = ""
                row_data["context_from_db"] = context
                
            with Timer() as t:
                try:
                    llm_res, _ = query_llm(model_url, question, context, imgs)
                    row_data["answer_from_model"] = llm_res
                except Exception as e:
                    _log.error(f"Answer generation failed: {str(e)}")
                    llm_res = ""
                row_data["answer_generation_time"] = t.seconds_from_start
                
            test_case = LLMTestCase(
                input=question,
                actual_output=llm_res,
                expected_output=correct_answer,
                context=[correct_context],
                retrieval_context=[context],
            )
            for metric in metrics_to_calculate:
                try:
                    metric.measure(test_case)
                    row_data[f"{metric.__name__}_score"] = metric.score
                    row_data[f"{metric.__name__}_reason"] = metric.reason
                except Exception as e:
                    row_data[f"{metric.__name__}_score"] = -1
                    row_data[f"{metric.__name__}_reason"] = f"{type(e).__name__}: {str(e)}"
            
            row_df = pd.DataFrame([row_data])
            with open(path_to_df, 'a', newline='') as f:
                row_df.to_csv(f, header=f.tell() == 0, index=False)
        
        except Exception as e:
            _log.error(f"Critical error processing question {i}: {str(e)}")
            if 'row_df' in locals():
                with open(path_to_df, 'a', newline='') as f:
                    row_df.to_csv(f, header=f.tell() == 0, index=False)
            raise
        
    result_df = pd.read_csv(path_to_df)
    result_df["total_time"] = (
            result_df["context_retrieve_time"] + result_df["answer_generation_time"]
    )
    # Calculation of basic statistics for exec time and function selection
    avg_context_retrieve_time = result_df["context_retrieve_time"].mean().round(2)
    avg_ans_generation_time = result_df["answer_generation_time"].mean().round(2)
    avg_total_time = result_df["total_time"].mean().round(2)
    # Calculation of statistics for metrics
    metrics_score_columns = list(filter(lambda x: "score" in x, result_df.columns.tolist()))
    metrics_to_print = []
    for column in metrics_score_columns:
        result_df[column] = pd.to_numeric(result_df[column])
        avg_score = result_df[result_df[column] != -1][column].mean()
        failed_evaluations = result_df[result_df[column] == -1].shape[0]
        metrics_to_print.append(
            f"- Average {column} is {avg_score}. Number of unsuccessfully processed questions {failed_evaluations}"
        )
    short_metrics_result = "\n".join(metrics_to_print)
    
    to_print = f"""Average context retrieving time: {avg_context_retrieve_time}
Average answer generation time: {avg_ans_generation_time}
Average total time: {avg_total_time}
Short metrics results:
{short_metrics_result}"""
    
    with open(path_to_results, "w") as f:
        print(to_print, file=f)
    
    return result_df


if __name__ == "__main__":
    model_1 = "gpt-4o-mini"
    model_2 = "https://api.vsegpt.ru/v1;openai/gpt-4o-mini"
    v = 0.1
    pipeline_test_with_save(all_questions,[correctness_metric], model_1, model_2, v)