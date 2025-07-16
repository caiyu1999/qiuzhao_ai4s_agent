from Config import Config
from Graph.Graph_state import init_graph_state


if __name__ == "__main__":
    config = Config.from_yaml("test/test_config.yaml")
    graph_state = init_graph_state(config)
    print(graph_state)
    
    