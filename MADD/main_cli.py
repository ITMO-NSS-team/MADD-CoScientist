from definitions import CONFIG_PATH
from dotenv import load_dotenv

load_dotenv(CONFIG_PATH)

from MADD.mas.create_graph import create_by_default_setup


if __name__ == "__main__":
    graph = create_by_default_setup()
    for step in graph.stream({'input': input()}, user_id="1"):
        print(step)
