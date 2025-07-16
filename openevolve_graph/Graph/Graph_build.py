''' 
这个文件实现一个graph_builder 根据config的文件生成graph
'''
from openevolve_graph.Config import Config
from typing import List 
from langchain.chat_models import init_chat_model 
from dataclasses import dataclass 




















class GraphBuilder:
    def __init__(self,
                 config:Config):
        
        
        self.config = config
        self.llms = config.llm
        # self.evaluator = config.evaluator
        
        self.llm_models = [init_chat_model(**{k:v for k,v in model.to_dict().items()}) for model in self.llms.models]
        self.evaluator_models = [init_chat_model(**{k:v for k,v in model.to_dict().items()}) for model in self.llms.evaluator_models]

    def build_graph(self):
        '''
        根据config文件构建图结构 如果岛屿为n个 则构建n个子图  每一个子图内部迭代n次后等待其他子图完成 然后进行迁移 
        ''' 
        pass
        
    def resume_from_checkpoint(self):
        pass 
         
    def build_subgraph(self):
        pass 
        
    def add_node(self):
        pass 


        
        
        
        
        
        
    
    
if __name__ == "__main__":
    config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
    graph_builder = GraphBuilder(config)
    graph_builder.build_graph()