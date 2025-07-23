from openevolve_graph.Graph.Graph_state import GraphState
from openevolve_graph.Config import Config 
from openevolve_graph.Graph.Graph_state import IslandState
import time
import logging
logger = logging.getLogger(__name__)





class routing_llm_generate_successful(object):
    '''
    
    '''
    def __init__(self,config:Config,island_id:str):
        self.config = config
        self.island_id = island_id
    def __call__(self,state:IslandState) -> bool:
        if state.llm_generate_success == True:#证明模型正常生成了答案 
            
            return True #跳转到下一个节点 
        else:
            return False #跳转到原节点

class routing_evaluate_successful(object):
    '''
    
    '''
    def __init__(self,config:Config,island_id:str):
        self.config = config
        self.island_id = island_id
    def __call__(self,state:IslandState) -> bool:
        if state.evaluate_success == True:#证明模型正常生成了答案 
            
            return True #跳转到下一个节点 
        else:
            return False #跳转到原节点

class routing_iteration_end(object):
    '''
    这个routing负责判定是否到达了迭代次数
    
    '''
    
    def __init__(self,config:Config,island_id:str,meeting_interval:int):
        self.config = config
        self.island_id = island_id
        self.meeting_interval = meeting_interval
    def __call__(self,state:IslandState) -> bool:
        if state.iteration%self.meeting_interval == 0:
            logger.info(f"Island:{state.id} has reached the time of meeting")
            return True
        else:
            logger.info(f"Island:{state.id} now start iteration: {state.iteration}")
            return False