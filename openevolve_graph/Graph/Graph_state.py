from pydantic import BaseModel,Field 
from abc import abstractmethod 
from typing import Dict,Any,Optional,List,Union
from dataclasses import asdict,dataclass
from program import Program
from openevolve_graph.Config import Config
from openevolve_graph.utils.utils import load_initial_program
import uuid 


class GraphState(BaseModel):
    ''' 
    图的共享状态 在运行的过程中会不断更新
    '''
    
    init_program:str = Field(default="") # 初始程序的code 全局只有一个
    best_program:Optional[str] = Field(default="") # 最好的程序id 全局只有一个
    best_program_id:str = Field(default="") # 最好的程序的id 全局只有一个
    best_metrics:Optional[Dict[str,Any]] = Field(default_factory=dict) # 最好的程序的指标 
    top_programs:List[str] = Field(default_factory=list) # 几个最好的程序  里面只存放id
    generation_count:List[int] = Field(default_factory=list) # 每一个岛屿当前的代数 
    num_islands:int = Field(default=0) # 岛屿的数量 
    island_programs: List[List[str]] = Field(default_factory=list) # 各个岛屿上的程序id
    all_programs: Dict[str,Any] = Field(default_factory=dict) # 所有的程序 dict{"program_id":code}
    status:List[str] = Field(default_factory=list) # 岛屿当前的状态 (运行到了哪一步)
    def to_dict(self)->Dict[str,Any]:
        return self.model_dump()
    
    def from_dict(self,data:Dict[str,Any])->"GraphState":
        return GraphState(**data)
def init_graph_state(config:Config)->GraphState:
    '''
    初始化图的状态
    '''
    if config.init_program_path is None:
        raise ValueError("init_program is not set")
    init_program = load_initial_program(config.init_program_path)
    num_islands = config.island.num_islands
    generation_count = [0]*num_islands
    initial_program_id = str(uuid.uuid4())
    island_programs = [[initial_program_id] for _ in range(num_islands)]
    all_programs = {initial_program_id:init_program}
    best_metrics = {}
    
    
    
    
    
    
if __name__ == "__main__":
    config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
    graph_state = init_graph_state(config)
    print(graph_state)
    
    
        
    
    
    
    
    
    
    