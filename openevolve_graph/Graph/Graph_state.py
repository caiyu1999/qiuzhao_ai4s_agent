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


def reducer_tuple(left: Dict[str, Any], right: Optional[Dict[str,Any]]|Tuple[str,Any]|List[Tuple[str,Any]]|None):
    '''
    在初始化时 传入的right和left是一个类型  用right替换left即可
    在运行过程中 传入的right是tuple类型 需要将tuple中的值更新到left中
    '''
    print(f"reducer_tuple left: {left}, right: {right}")
    if right is None:
        return left 
    
    if isinstance(right,dict):
        print(f"reducer_tuple right is a dict")
        return right
    
    elif isinstance(right,list):
        merge = left.copy()
        for item in right:
            if len(item) == 2:
                island_id = item[0]
                value = item[1]
                merge[island_id] = value
            else:
                raise ValueError("reducer_tuple right must be a tuple of length 2, but you give me a {}".format(len(item)))
    elif isinstance(right,tuple):
        merge = left.copy()
        if len(right) == 2:
            island_id = right[0]
            value = right[1]
            merge[island_id] = value
        else:       
            raise ValueError("reducer_tuple right must be a tuple of length 2 or 3, but you give me a {}".format(len(right)))
    return merge 


def reducer_for_safe_container_island_programs(left:Dict[str,ThreadSafePrograms],right:Optional[Tuple[str,str,Program]|Tuple[str,str,str,str,Program]| Dict[str,ThreadSafePrograms]]|List[Tuple[str,Any]])->Dict[str,ThreadSafePrograms]:
    '''
    这里需要定义新的程序添加时的更新方式 
    '''
    print(f"reducer_for_safe_container_island_programs left: {left}, right: {right}")
    if right is None:
        return left 
    
    if isinstance(right,dict):
        return right
    
    elif isinstance(right,list): #添加，删除，更新多个程序 或者 替换多个程序
        merge = left.copy()
        for item in right:
            if len(item) == 3:
                operation = item[0]
                island_id = item[1]
                program = item[2]
                if operation == "add":
                    merge[island_id].add_program(program.id,program)
                elif operation == "remove":
                    merge[island_id].remove_program(program.id)
                elif operation == "update":
                    merge[island_id].update_program(program.id,program)
                else:
                    raise ValueError("reducer_for_safe_container_island_programs right must be a tuple of length 3, but you give me a {}".format(len(item)))
            if len(item) == 5:
                operation = item[0]
                island_id = item[1]
                program_need_replace_id = item[2]# 需要被替换的程序id
                program_replace_with_id = item[3]# 替换的程序id
                program_replace_with_program = item[4]# 替换的程序对象
                if operation == 'replace':
                    merge[island_id].remove_program(program_need_replace_id)
                    merge[island_id].add_program(program_replace_with_id,program_replace_with_program)
                else:
                    raise ValueError("reducer_for_safe_container_island_programs right must be a tuple of length 4, but you give me a {}".format(len(item)))
        return merge 
    
    
    elif isinstance(right,tuple): #添加，删除，更新单个程序 或者 替换单个程序
        if len(right) == 3:
            merge = left.copy()
            operation = right[0]
            island_id = right[1]
            program = right[2]
            if operation == "add":
                merge[island_id].add_program(program.id,program)    
            elif operation == "remove":
                merge[island_id].remove_program(program.id)
            elif operation == "update":
                merge[island_id].update_program(program.id,program)
            else:
                raise ValueError("reducer_for_safe_container_island_programs right must be a tuple of length 3, but you give me a {}".format(len(right)))
            return merge 
        if len(right) == 5:
            merge = left.copy()
            operation = right[0]
            island_id = right[1]
            program_need_replace_id = right[2]# 需要被替换的程序id
            program_replace_with_id = right[3]# 替换的程序id
            program_replace_with_program = right[4]# 替换的程序对象
            if operation == 'replace':
                merge[island_id].remove_program(program_need_replace_id)
                merge[island_id].add_program(program_replace_with_id,program_replace_with_program)
            else:
                raise ValueError("reducer_for_safe_container_island_programs right must be a tuple of length 4, but you give me a {}".format(len(item)))
    


def reducer_for_safe_container_all_programs(left:ThreadSafePrograms,right:Optional[Tuple[str,Program]|ThreadSafePrograms]|List[Tuple[str,Any]])->ThreadSafePrograms:
    '''
    这里需要定义新的程序添加时的更新方式 
    '''
    print(f"reducer_for_safe_container_all_programs left: {left}, right: {right}")
    if right is None:
        print(f"reducer_for_safe_container_all_programs right is None")
        return left 
    
    if isinstance(right,ThreadSafePrograms):
        return right
    
    elif isinstance(right,list):
        merge = left.copy()
        for item in right:
            if len(item) == 2:
                operation = item[0]
                program = item[1]
                if operation == "add":
                    merge.add_program(program.id,program)
                elif operation == "remove":
                    merge.remove_program(program.id)
                elif operation == "update":
                    merge.update_program(program.id,program)
                else:
                    raise ValueError("reducer_for_safe_container_all_programs right must be a tuple of length 2, but you give me a {}".format(len(item)))
        return merge 
    






class IslandStatus(Enum):
    '''
    每个岛屿的状态，
    这个状态表示刚刚结束时最后运行的节点
    '''
    INIT_STATE = "init_state" # 初始化状态 
    INIT_EVALUATE = "init_evaluate" # 初始化评估 
    INIT = "init" # 初始化这个岛屿 可能是读取checkpoint 也可能是从初始程序开始
    SAMPLE = "sample" # 采样父代程序与灵感程序
    GET_ARTIFACTS = "get_artifacts" # 获取工件
    GET_TOP_PROGRAMS = "get_top_programs" # 获取最好的程序
    BUILD_PROMPT = "build_prompt" # 构建提示词
    LLM_GENERATE = "llm_generate" # 使用llm生成程序
    GENERATE_CHILD = "generate_child" # 生成子代程序 
    EVALUATE_CHILD = "evaluate_child" # 评估子代程序 并添加到程序库中
    UPDATE = "update" # 更新岛屿 精英程序等信息 
    
    
    # APPLY_DIFF = "apply_diff" # 应用差异
    # EVALUTE_PROGRAM = "evalute_program" # 评估程序
    # GET_PENDING_ARTIFACTS = "get_pending_artifacts" # 获取待处理的工件
    # ADD_PROGRAM = "add_program" # 将程序存入数据库
    # MEETING = "meeting" # 交流会 在交流会会进行迁移

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


    best_program:str = Field(default="") # 最好的程序code 全局只有一个 随时更新 默认reducer
    best_program_id:str = Field(default="") # 最好的程序的id 全局只有一个 随时更新 默认reducer
    best_metrics:Dict[str,Any] = Field(default_factory=dict) # 最好的程序的指标 随时更新 默认reducer

    best_program_each_island:Annotated[Dict[str,str],reducer_tuple] = Field(default_factory=dict) # 每个岛屿上最好的程序id  岛屿内部更新 安全 e.g. {"island_id":["program_id1","program_id2"]}
    generation_count:Annotated[Dict[str,int],reducer_tuple] = Field(default_factory=dict) # 每一个岛屿当前的代数  岛屿内部更新 安全 e.g. {"island_id":0}
    archive:Annotated[ThreadSafePrograms,reducer_for_safe_container_all_programs] = Field(default_factory=ThreadSafePrograms) # 精英归档 里面存放Program对象 各个岛屿上的精英归档汇总在这个dict中 随时更新

    island_programs: Annotated[Dict[str,ThreadSafePrograms],reducer_for_safe_container_island_programs] = Field(default_factory=dict) # 各个岛屿上的程序id 岛屿内部更新 安全 e.g. {"island_id":["program_id1","program_id2"]}
    # 使用线程安全的程序管理器
    all_programs: Annotated[ThreadSafePrograms, reducer_for_safe_container_all_programs] = Field(default_factory=ThreadSafePrograms)
    newest_programs: Annotated[Dict[str, str],reducer_tuple] = Field(default_factory=dict) # 各个岛屿上最新的程序id 岛屿内部更新 安全 e.g. {"island_id":"program_id"}
    island_generation_count:Annotated[Dict[str,int],reducer_tuple] = Field(default_factory=dict) # 各个岛屿当前的代数 岛屿内部更新 安全 e.g. {"island_id":0}

    #交流会相关
    generation_count_in_meeting:int = Field(default=0) # 交流会进行的次数
    time_of_meeting:int = Field(default=10) # 每当各个岛屿迭代了time_of_meeting次 就会进行一次交流会 安全

    #岛屿内部进化有关:
    current_program_id: Annotated [Dict[str,str],reducer_tuple] = Field(default_factory=dict) # 各个岛屿上当前的程序id(child id) 岛屿内部更新 安全 e.g. {"island_id":"program_id"}
    current_program_code: Annotated[Dict[str,str],reducer_tuple] = Field(default_factory=dict) # 各个岛屿上当前的程序code 岛屿内部更新 安全 e.g. {"island_id":"program_code"}
    sample_program_id: Annotated[Dict[str,str], reducer_tuple] = Field(default_factory=dict) # 各个岛屿上采样的父代程序id 岛屿内部更新 安全 e.g. {"island_id":"program_id"}
    sample_inspirations: Annotated[Dict[str,List[str]], reducer_tuple] = Field(default_factory=dict) # 各个岛屿上采样的程序的灵感程序id 岛屿内部更新 安全 e.g. {"island_id":["program_id1","program_id2"]}
    prompt:Annotated[Dict[str,str],reducer_tuple]=Field(default_factory=dict) # 各个岛屿上用于构建的提示词 岛屿内部更新 安全 e.g. {"island_id":"prompt"}
    sample_top_programs:Annotated[Dict[str,List[str]],reducer_tuple]=Field(default_factory=dict) # 各个岛屿上采样的最好的程序id 岛屿内部更新 安全 e.g. {"island_id":["program_id1","program_id2"]}
    feature_map: Annotated[Dict[str,Any], reducer_tuple] = Field(default_factory=dict) # 全部程序中的特征坐标 全局更新 e.g{"program_id":[0,1,2,-1]}


    llm_generate_success:Annotated[Dict[str,bool],reducer_tuple]=Field(default_factory=dict) # 各个岛屿上llm生成是否成功 岛屿内部更新 安全 e.g. {"island_id":True}
    llm_message_diff:Annotated[Dict[str,str],reducer_tuple]=Field(default_factory=dict) # llm返回的修改代码的建议
    llm_message_rewrite:Annotated[Dict[str,str],reducer_tuple]=Field(default_factory=dict) # llm返回的修改代码的重写部分  
    llm_message_suggestion:Annotated[Dict[str,str],reducer_tuple]=Field(default_factory=dict) # llm返回的修改代码的diff部分
    llm_change_summary:Annotated[Dict[str,str],reducer_tuple]=Field(default_factory=dict) # 修改代码的总结 
    
    
    lock : bool = Field(default=False) # 是否锁定  在上锁时无法对state进行更新
    
    

    def to_dict(self)->Dict[str,Any]:
        return self.model_dump()

    def from_dict(self,data:Dict[str,Any])->"GraphState":
        return GraphState(**data)
    
    
        







if __name__ == "__main__":
    config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
