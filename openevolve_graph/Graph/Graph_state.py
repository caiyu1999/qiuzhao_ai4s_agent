from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Annotated , Tuple
from openevolve_graph.program import Program
from openevolve_graph.Config import Config

import uuid
import os
import time
import asyncio
from enum import Enum
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing_extensions import Annotated
# 合并函数现在从 langgraph_compatible_container 模块导入
from openevolve_graph.utils.thread_safe_programs import ThreadSafePrograms

# 合并函数现在从 thread_safe_container 模块导入


def reducer_tuple(left: Dict[str, Any], right: Optional[Dict[str,Any]]|Tuple[str,Any]|Tuple[str,str,Any]|None):
    '''
    在初始化时 传入的right和left是一个类型  用right替换left即可
    在运行过程中 传入的right是tuple类型 需要将tuple中的值更新到left中
    '''
    
    if right is None:
        return left 
    
    if isinstance(right,dict):
        return {**left,**right}
    elif isinstance(right,tuple):
        if len(right) == 2:
            island_id = right[0]
            value = right[1]
            left[island_id] = value
        elif len(right) == 3:
            pass
        else:
            raise ValueError("reducer_tuple right must be a tuple of length 2 or 3, but you give me a {}".format(len(right)))
    return left 
#     """
#     更新部分岛屿内部信息的reducer函数
    
#     适用于以下更新形式
    
#     e.g
    
#     "sample_program_id": {self.island_id: parent_id},
#     "sample_inspirations": {self.island_id: inspirations},
#     "status":{self.island_id:IslandStatus.SAMPLE}
    
#     某个属性是一个字典 传入的也是一个字典 将传入字典中的键值更新到left中 其他不变
#     """
    
#     # 确保 left 是字典 - 处理 None 或其他类型
#     if not isinstance(left, dict):
#         raise ValueError("left must be a dict, but you give me a {}".format(type(left)))
    
#     # 确保 right 是字典 - 处理 None 或其他类型  
#     if not isinstance(right, tuple):
#         if isinstance(right,type(left)):
#             return right
        
#         raise ValueError("right must be a tuple, but you give me a {},the right is {}".format(type(right),right))
#     # 创建副本以避免修改原始数据
#     merged = left.copy() 
    
    
#     #将传入的键值更新到merged中
    
#     island_id = right[0]
#     value = right[1]
    
#     merged[island_id] = value
    
#     return merged

def reducer_for_safe_container(left:ThreadSafePrograms,right:ThreadSafePrograms)->ThreadSafePrograms:
    # '''
    # Tuple right : (move_type,program_id,program)
    # move_type:str : delete or add 
    # program_id:str : program id
    # program:Program : program object
    # '''
    # if not isinstance(left,ThreadSafePrograms):
    #     raise ValueError("left must be a ThreadSafePrograms, but you give me a {}".format(type(left)))
    
    # if not isinstance(right,tuple):
    #     raise ValueError("right must be a tuple, but you give me a {},the right is {}".format(type(right),right))
    
    # if len(right)!=3:
    #     raise ValueError("reducer for safe container right must be a tuple of length 3, but you give me a {},the right is {}".format(len(right),right))
    
    # move_type = right[0]
    # program_id = right[1]
    # program = right[2]
    
    # if move_type == "delete":
    #     left.remove_program(program_id)
    # elif move_type == "add":
    #     left.add_program(program_id,program)
    
    return right


class IslandStatus(Enum):
    '''
    每个岛屿的状态，
    这个状态表示刚刚结束时最后运行的节点
    '''
    INIT_STATE = "init_state" # 初始化状态 这个状态表示刚刚结束时最后运行的节点
    INIT = "init" # 初始化这个岛屿 可能是读取checkpoint 也可能是从初始程序开始
    SAMPLE = "sample" # 采样父代程序与灵感程序
    GET_ARTIFACTS = "get_artifacts" # 获取工件
    GET_TOP_PROGRAMS = "get_top_programs" # 获取最好的程序
    BUILD_PROMPT = "build_prompt" # 构建提示词
    LLM_GENERATE = "llm_generate" # 使用llm生成程序
    APPLY_DIFF = "apply_diff" # 应用差异
    EVALUTE_PROGRAM = "evalute_program" # 评估程序
    GET_PENDING_ARTIFACTS = "get_pending_artifacts" # 获取待处理的工件
    ADD_PROGRAM = "add_program" # 将程序存入数据库
    MEETING = "meeting" # 交流会 在交流会会进行迁移

    # 迁移



class GraphState(BaseModel):
    '''
    图的共享状态 在运行的过程中会不断更新
    要注意在并行运行的时候不同的子图会同时修改某个状态 要保证不能冲突
    只有在岛屿交流会议上 才会更新一些共享内容 其他时候都是岛屿内部更新 安全
    
    需要指定里面所有值的更新方式
'''       
    # 配置 Pydantic 以允许任意类型
    model_config = {"arbitrary_types_allowed": True}
    # 全局共享状态
    init_program:str = Field(default="") # 初始程序的id 全局只有一个 这个值不会被更新 安全
    evaluation_program:str = Field(default="") # 评估程序的code 全局只有一个 这个值不会被更新 安全
    language:str = Field(default="python") # 编程语言 安全
    file_extension:str = Field(default="py") # 文件扩展名 安全
    num_islands:int = Field(default=0) # 岛屿的数量  安全
    islands_id:List[str] = Field(default_factory=list) # 岛屿的id 安全 岛屿的id是全局唯一的 不会被更新 安全
    status:Annotated[Dict[str,Any],reducer_tuple] = Field(default_factory=dict) # 岛屿当前的状态 (运行到了哪一步) 岛屿内部更新 安全 e.g. {"island_id":IslandStatus.INIT} 默认reducer
    # island_evolution_direction:Dict[str,str] = Field(default_factory=dict) # 岛屿的进化方向 这里暂定空字典 在后面添加 负责指导岛屿的总体进化方向 安全 e.g. {"island_id":"evolution_direction"}


    best_program:Optional[str | None] = Field(default=None) # 最好的程序code 全局只有一个 随时更新 默认reducer
    best_program_id:str = Field(default="") # 最好的程序的id 全局只有一个 随时更新 默认reducer
    best_metrics:Optional[Dict[str,Any]| None] = Field(default_factory=dict) # 最好的程序的指标 随时更新 默认reducer

    best_program_each_island:Annotated[Dict[str,str],reducer_tuple] = Field(default_factory=dict) # 每个岛屿上最好的程序id  岛屿内部更新 安全 e.g. {"island_id":["program_id1","program_id2"]}
    generation_count:Annotated[Dict[str,int],reducer_tuple] = Field(default_factory=dict) # 每一个岛屿当前的代数  岛屿内部更新 安全 e.g. {"island_id":0}
    archive:Annotated[ThreadSafePrograms,reducer_for_safe_container] = Field(default_factory=ThreadSafePrograms) # 精英归档 里面存放Program对象 各个岛屿上的精英归档汇总在这个dict中 随时更新

    island_programs: Annotated[Dict[str,List[str]],reducer_tuple] = Field(default_factory=dict) # 各个岛屿上的程序id 岛屿内部更新 安全 e.g. {"island_id":["program_id1","program_id2"]}
    # 使用线程安全的程序管理器
    all_programs: Annotated[ThreadSafePrograms, reducer_for_safe_container] = Field(default_factory=ThreadSafePrograms)
    newest_programs: Annotated[Dict[str, str],reducer_tuple] = Field(default_factory=dict) # 各个岛屿上最新的程序id 岛屿内部更新 安全 e.g. {"island_id":"program_id"}
    island_generation_count:Annotated[Dict[str,int],reducer_tuple] = Field(default_factory=dict) # 各个岛屿当前的代数 岛屿内部更新 安全 e.g. {"island_id":0}

    #交流会相关
    generation_count_in_meeting:int = Field(default=0) # 交流会进行的次数
    time_of_meeting:int = Field(default=10) # 每当各个岛屿迭代了time_of_meeting次 就会进行一次交流会 安全

    #岛屿内部进化有关:
    current_program_id: Annotated [Dict[str,str],reducer_tuple] = Field(default_factory=dict) # 各个岛屿上当前的程序id(child id) 岛屿内部更新 安全 e.g. {"island_id":"program_id"}
    sample_program_id: Annotated[Dict[str,str], reducer_tuple] = Field(default_factory=dict) # 各个岛屿上采样的父代程序id 岛屿内部更新 安全 e.g. {"island_id":"program_id"}
    sample_inspirations: Annotated[Dict[str,List[str]], reducer_tuple] = Field(default_factory=dict) # 各个岛屿上采样的程序的灵感程序id 岛屿内部更新 安全 e.g. {"island_id":["program_id1","program_id2"]}
    artifacts:Annotated[Dict[str,Any],reducer_tuple]=Field(default_factory=dict) #从采样的父代程序得到的工件
    prompt:Annotated[Dict[str,str],reducer_tuple]=Field(default_factory=dict) # 各个岛屿上构建的提示词 岛屿内部更新 安全 e.g. {"island_id":"prompt"}
    sample_top_programs:Annotated[Dict[str,List[str]],reducer_tuple]=Field(default_factory=dict) # 各个岛屿上采样的最好的程序id 岛屿内部更新 安全 e.g. {"island_id":["program_id1","program_id2"]}
    feature_map: Annotated[Dict[str,Any], reducer_tuple] = Field(default_factory=dict) # 各个岛屿上的特征 岛屿内部更新 安全 e.g. {"island_id":{"feature_name":feature_value}}

    
        


    def to_dict(self)->Dict[str,Any]:
        return self.model_dump()

    def from_dict(self,data:Dict[str,Any])->"GraphState":
        return GraphState(**data)
    
    
        







if __name__ == "__main__":
    config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
    graph_state = init_graph_state(config)
    # print(graph_state.all_programs)
    print(graph_state)
   
    