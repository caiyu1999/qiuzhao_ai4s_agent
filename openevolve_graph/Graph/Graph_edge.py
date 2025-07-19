from openevolve_graph.Graph.Graph_state import GraphState
from openevolve_graph.Config import Config 
from openevolve_graph.Graph.Graph_state import IslandStatus





class routing_edge_llm_generate_failed(object):
    '''
    这个edge是当LLM生成失败时 跳转 跳转的选项可以为 原节点 采样节点 生成子代节点
    '''
    def __init__(self,config:Config,island_id:str):
        self.config = config
        self.island_id = island_id
    def __call__(self,state:GraphState) -> bool:
        if state.status[self.island_id] == IslandStatus.LLM_GENERATE:#证明模型正常生成了答案 
            return True #跳转到下一个节点 
        else:
            return False #跳转到原节点
