import asyncio 
from openevolve_graph.Config import Config,ModelConfig
from openevolve_graph.Graph import GraphBuilder
import evaluator
from openevolve_graph.Graph.Graph_state import GraphState


async def main(config_path:str):
    
    
    # 1. load config 
    config = Config.from_yaml(config_path)
    
    # 2. save config as dict for further use and graph state 
    config_dict = config.to_dict()
    
    llms = [ModelConfig(**llm) for llm in config_dict["llm"]["models"]]
    
    evaluator_models = [ModelConfig(**model) for model in config_dict["llm"]["evaluator_models"]]
    
    graph_builder = GraphBuilder(llms,evaluator_models,config)
    
    graph_state = GraphState()
    print(graph_state.init_program)
    
  
    
    
    







    
    
    
    
    
    
     


if __name__ == "__main__":
    asyncio.run(main("/Users/caiyu/Desktop/langchain/openevolve_graph/test/test_config.yaml"))