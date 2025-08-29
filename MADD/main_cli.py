from dotenv import load_dotenv
load_dotenv("config.env")

from MADD.mas.create_graph import create_by_default_setup

if __name__ == "__main__":
    graph = create_by_default_setup()
    for step in graph.stream(
        {
            "input": "Text me all available predictive models"
        },
        user_id="1",
    ):
        print(step)
