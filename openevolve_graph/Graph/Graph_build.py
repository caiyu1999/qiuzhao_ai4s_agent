''' 
这个文件实现一个graph_builder 根据config的文件生成graph
'''
from Config import ModelConfig,Config
from typing import List 
from langchain.chat_models import init_chat_model 
from dataclasses import dataclass 




















class GraphBuilder:
    def __init__(self,
                 llms:List[ModelConfig],
                 evaluator_models:List[ModelConfig],
                 config:Config):
        
        self.llms = llms
        self.evaluator_models = evaluator_models
        
        self.llm_models = [init_chat_model(**{k:v for k,v in model.to_dict().items()}) for model in llms]
        self.evaluator_models = [init_chat_model(**{k:v for k,v in model.to_dict().items()}) for model in evaluator_models]
        
        self.checkpoint_path = config.checkpoint_path
        

    
    
    
    def build_graph(self):
        pass 
        
    def resume_from_checkpoint(self):
        pass 
        
        



        
        
        
        
        
        
    