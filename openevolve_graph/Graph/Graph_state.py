from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Annotated , Tuple
from openevolve_graph.program import Program
from openevolve_graph.Config import Config
import operator 
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
from openevolve_graph.utils.utils import _is_better
# 合并函数现在从 langgraph_compatible_container 模块导入
from openevolve_graph.utils.thread_safe_programs import Programs_container
import json 
import logging
logger = logging.getLogger(__name__)
from openevolve_graph.Graph.RAG_document import document

def reducer_for_feature_map(left:Dict[str,Any],
                            right:Optional[Tuple[str,str]]|Dict[str,Any]|Tuple[str,str,str])->Dict[str,Any]|None:
    if right is None:
        return left 
    
    if isinstance(right,dict):
        return right 
    
    elif isinstance(right,tuple):
        if len(right) == 2:
            feature_key = right[0]
            program_id = right[1]
            left[feature_key] = program_id
    else:
        raise ValueError("reducer_for_feature_map right must be a tuple of length 2, but you give me a {}".format(len(right)))
    return left

def reducer_for_safe_container_all_programs(left:Programs_container,right:Optional[Tuple[str,Program]|Programs_container]|List[Tuple[str,Any]]|Tuple[str,str,Program])->Programs_container:
    '''
    这里需要定义新的程序添加时的更新方式 
    '''
    #print(f"reducer_for_safe_container_all_programs left: {left}, right: {right}")
    if right is None:
        #print(f"reducer_for_safe_container_all_programs right is None")
        return left 
    
    if isinstance(right,Programs_container):
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
    
    elif isinstance(right,tuple):
        merge = left.copy()
        if len(right) == 2:
            operation = right[0]
            program = right[1]
            if operation == "add":
                merge.add_program(program.id,program)
            elif operation == "remove":
                merge.remove_program(program.id)
            elif operation == "update":
                merge.update_program(program.id,program)
            else:
                raise ValueError("reducer_for_safe_container_all_programs right must be a tuple of length 2, but you give me a {}".format(len(right)))
        
        elif len(right) == 3:
            operation = right[0]
            program_need_replace_id = right[1] #将被替换的程序id
            program_replace_with_program = right[2] #替换的程序对象
            if operation == "replace":
                merge.remove_program(program_need_replace_id)
                merge.add_program(program_replace_with_program.id,program_replace_with_program)
            else:
                raise ValueError("reducer_for_safe_container_all_programs right must be a tuple of length 4, but you give me a {}".format(len(right)))
        else:
            raise ValueError("reducer_for_safe_container_all_programs right must be a tuple of length 2 or 4, but you give me a {}".format(len(right)))
        return merge 
    
    
    
    
def reducer_best_program(left:Program,right:Program)->Program:
    '''
    best_program的更新方式 只需传入最好的Program即可
    '''
    
    if not isinstance(right,Program):
        raise ValueError(f"right is not a Program object: {right}")
    if not isinstance(left,Program):
        raise ValueError(f"left is not a Program object: {left}")
    if _is_better(right,left):
        return right 
    return left


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


def reducer_for_single_parameter(left:Any,right:Any)->Any:
    '''
    这个函数用来更新单个参数  这个参数在迭代过程中是永远不变的 
    '''
    return right 
    
class IslandState(BaseModel):
    '''
    对于一个岛屿来说 同一个时间只有一个节点能够修改它的value 所以不用担心线程安全问题
    
    '''
    
    model_config = {"arbitrary_types_allowed": True}
    # 岛屿id
    id:str = Field(default=f"{__name__}")
    # 岛屿上的所有程序 存放在这个container中
    programs:Annotated[Programs_container,reducer_for_safe_container_all_programs] = Field(default_factory=Programs_container)
    # 岛屿上目前的最新程序(child) 
    latest_program: Program = Field(default_factory=Program)
    # 当前的状态(最后运行的节点)
    status:IslandStatus = Field(default=IslandStatus.INIT_STATE)
    # 生成的提示词  
    prompt:str = Field(default="")
    
    language:str = Field(default="python")
    
    # sample_program 采样的父代程序
    sample_program:Program = Field(default_factory=Program)
    # sample_inspirations 采样的灵感程序 
    sample_inspirations:List[str] = Field(default_factory=list)
    # 岛屿上最好的程序
    best_program:Program = Field(default_factory=Program)
    
    # iteration 总迭代次数  
    iteration:int = Field(default=0)
    
    #距离上次meeting后的迭代次数
    now_meeting:int = Field(default=0)
    #距离下次meeting的迭代次数
    next_meeting:int = Field(default=0)
    
    
    # 岛屿的进化方向 这里暂定空字典 在后面添加 负责指导岛屿的总体进化方向 安全 e.g. {"island_id":"evolution_direction"}
    # island_evolution_direction:str = Field(default="") #这个在后期可以实现
    
    # LLM_GENERATE 相关
    # 在基于diff进化的情况下启用 
    diff_message:str = Field(default="")
    # 在基于rewrite进化的情况下启用 
    rewrite_message:str = Field(default="")
    # 改进代码的suggestion 
    suggestion_message:str = Field(default="")
    # 总结 
    change_summary:str = Field(default="")
    
    llm_generate_success:bool = Field(default=False)
    evaluate_success:bool = Field(default=False)
    
    # 每次meeting后分配给各个岛屿的临时文件 
    # "all programs" 全部程序 实际上是主图中的程序库 岛屿内部会临时更新 但是在下一次meeting这个值会更新为最新的主图中的程序库
    all_programs:Annotated[Programs_container,reducer_for_safe_container_all_programs] = Field(default_factory=Programs_container) 
    # 全部程序的特征坐标 同上
    feature_map:Annotated[Dict[str,Any],reducer_for_feature_map] = Field(default_factory=dict)
    # 精英归档 里面存放Program对象 各个岛屿上的精英归档汇总在这个dict中  同上
    archive:Annotated[Programs_container,reducer_for_safe_container_all_programs] = Field(default_factory=Programs_container)
    # 全部程序中的最好程序
    all_best_program:Program = Field(default_factory=Program)
    
    RAG_help_info:str = Field(default="") # 用于帮助生成代码的RAG信息 在ragnode后更新
   
    

    
 
    def to_dict(self)->Dict[str,Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls,data:Dict[str,Any])->"IslandState":
        return IslandState(**data)

    def to_json(self)->str:
        # 使用Pydantic v2的model_dump_json方法，它会自动处理枚举序列化
        return self.model_dump_json(exclude_none=True, indent=2)
def reducer_IslandState(left:IslandState,right:IslandState)->IslandState:
    '''
    IslandState的更新方式 只需传入IslandState即可
    在汇合节点中 多个子图的IslandState会更新到父节点的状态中
    直接用新的覆盖旧的即可
    '''
    if not isinstance(right,IslandState):
        raise ValueError("reducer_IslandState right must be a IslandState, but you give me a {}".format(type(right)))
    if not isinstance(left,IslandState):
        raise ValueError("reducer_IslandState left must be a IslandState, but you give me a {}".format(type(left)))
    
    
    # 因为多个子图在汇合的时候
    return right
    
class GraphState(BaseModel):
    '''
    图的共享状态 在运行的过程中会不断更新
    要注意在并行运行的时候不同的子图会同时修改某个状态 要保证不能冲突
    只有在岛屿交流会议上 才会更新一些共享内容 其他时候都是岛屿内部更新 安全
    需要指定里面所有值的更新方式
    '''       
    # 配置 Pydantic 以允许任意类型
    model_config = {"arbitrary_types_allowed": True}
    
    iteration:int = Field(default=0)
    # 全局共享状态 
    # 对于const参数 就算是合并节点 也不会更新  
    init_program:Annotated[Program,reducer_for_single_parameter] = Field(default_factory=Program) # 初始程序的id 全局只有一个 这个值不会被更新 const
    
    
    evaluation_program:Annotated[str,reducer_for_single_parameter] = Field(default="") # 评估程序的code 全局只有一个 这个值不会被更新 const
    
    language:Annotated[str,reducer_for_single_parameter] = Field(default="python") # 编程语言 const
    
    file_extension:Annotated[str,reducer_for_single_parameter] = Field(default="py") # 文件扩展名 const
    
    num_islands:Annotated[int,reducer_for_single_parameter] = Field(default=0) # 岛屿的数量  const
    
    islands_id:Annotated[List[str],reducer_for_single_parameter] = Field(default_factory=list) # 岛屿的id const 岛屿的id是全局唯一的 不会被更新 
    
    best_program: Annotated[Program, reducer_best_program] = Field(default_factory=Program)
    
    #交流会相关
    generation_count_in_meeting:int = Field(default=0) # 交流会进行的次数
    
    

    
    # 以下内容在每一次meeting后更新 随即下放到每一个岛屿 
    
    # 精英归档 里面存放Program对象 各个岛屿上的精英归档汇总在这个dict中 随时更新
    archive:Annotated[Programs_container,reducer_for_safe_container_all_programs] = Field(default_factory=Programs_container) # 精英归档 里面存放Program对象 各个岛屿上的精英归档汇总在这个dict中 随时更新
    # 程序管理器
    all_programs: Annotated[Programs_container, reducer_for_safe_container_all_programs] = Field(default_factory=Programs_container)
    # 全部程序的特征坐标 
    feature_map:Annotated[Dict[str,Any],reducer_for_feature_map] = Field(default_factory=dict)
    
    islands:Annotated[Dict[str,IslandState],reducer_for_single_parameter] = Field(default_factory=dict)
    
    
    
    rag_doc_list:List[str] = Field(default_factory=list) # 用于RAG的文档存储位置 文件地址
    rag_doc_path:str = Field(default="") # 用于RAG的文档存储位置 文件夹地址
    
    Documents:Dict[str,document] = Field(default_factory=dict)# rag文档 里面存储了文档的id 和document对象
    vector_save_dir:str = Field(default="") # 矢量存储地址
    
    Documents_abstract:Dict[str,str] = Field(default_factory=dict) # 文档的摘要 大致记录了这个文档内部的主要内容 按照文档id存储
    
    RAG_help_info:str = Field(default="") # 用于帮助生成代码的RAG信息
    def to_dict(self)->Dict[str,Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls,data:Dict[str,Any])->"GraphState":
        return GraphState(**data)
    
    
    def to_json(self)->str:
        # 使用Pydantic v2的model_dump_json方法，它会自动处理枚举序列化
        return self.model_dump_json(exclude_none=True, indent=2)
    
if __name__ == "__main__":
    state = GraphState()
    print(state.to_json())