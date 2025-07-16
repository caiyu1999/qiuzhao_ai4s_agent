from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Annotated
from openevolve_graph.program import Program
from openevolve_graph.Config import Config
from openevolve_graph.utils.utils import load_initial_program, extract_code_language
from evaluator import direct_evaluate
import uuid
import os
import time
import asyncio
from enum import Enum 


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """合并两个字典，右边的值会覆盖左边的值"""
    if not left:
        left = {}
    if not right:
        right = {}
    
    merged = left.copy()
    merged.update(right)
    return merged


class IslandStatus(Enum):
    ''' 
    每个岛屿的状态，
    这个状态表示刚刚结束时最后运行的节点
    '''
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
    
    
    '''
    
    init_program:str = Field(default="") # 初始程序的id 全局只有一个 这个值不会被更新 安全
    evaluation_program:str = Field(default="") # 评估程序的code 全局只有一个 这个值不会被更新 安全
    language:str = Field(default="python") # 编程语言 安全
    file_extension:str = Field(default="py") # 文件扩展名 安全
    num_islands:int = Field(default=0) # 岛屿的数量  安全
    islands_id:List[str] = Field(default_factory=list) # 岛屿的id 安全 岛屿的id是全局唯一的 不会被更新 安全
    status:Dict[str,Any] = Field(default_factory=dict) # 岛屿当前的状态 (运行到了哪一步) 岛屿内部更新 安全 e.g. {"island_id":IslandStatus.INIT}
    # island_evolution_direction:Dict[str,str] = Field(default_factory=dict) # 岛屿的进化方向 这里暂定空字典 在后面添加 负责指导岛屿的总体进化方向 安全 e.g. {"island_id":"evolution_direction"}

    
    best_program:Optional[str] = Field(default="") # 最好的程序code 全局只有一个 在交流会更新
    best_program_id:str = Field(default="") # 最好的程序的id 全局只有一个 在交流会更新
    best_metrics:Optional[Dict[str,Any]] = Field(default_factory=dict) # 最好的程序的指标 在交流会更新

    best_program_each_island:Dict[str,str] = Field(default_factory=dict) # 每个岛屿上最好的程序id  岛屿内部更新 安全 e.g. {"island_id":["program_id1","program_id2"]}
    generation_count:Dict[str,int] = Field(default_factory=dict) # 每一个岛屿当前的代数  岛屿内部更新 安全 e.g. {"island_id":0}
    archive:List[str] = Field(default_factory=list) # 精英归档 里面存放id 各个岛屿上的精英归档汇总在这个list中 在交流会中更新
    
    island_programs: Dict[str,List[str]] = Field(default_factory=dict) # 各个岛屿上的程序id 岛屿内部更新 安全 e.g. {"island_id":["program_id1","program_id2"]}
    all_programs: Dict[str,Program] = Field(default_factory=dict) # 所有的程序 dict{"program_id":Program} 全局随时更新 但是在交流会外不允许删除 只允许添加 安全
    newest_programs:Dict[str,str] = Field(default_factory=dict) # 各个岛屿上最新的程序id 岛屿内部更新 安全 e.g. {"island_id":"program_id"}
    island_generation_count:Dict[str,int] = Field(default_factory=dict) # 各个岛屿当前的代数 岛屿内部更新 安全 e.g. {"island_id":0}
    
    #交流会相关
    generation_count_in_meeting:int = Field(default=0) # 交流会进行的次数
    time_of_meeting:int = Field(default=10) # 每当各个岛屿迭代了time_of_meeting次 就会进行一次交流会 安全
    
    #岛屿内部进化有关:
    sample_program_id: Annotated[Dict[str,str], merge_dicts] = Field(default_factory=dict) # 各个岛屿上采样的程序id 岛屿内部更新 安全 e.g. {"island_id":"program_id"}
    sample_inspirations: Annotated[Dict[str,List[str]], merge_dicts] = Field(default_factory=dict) # 各个岛屿上采样的程序的灵感 岛屿内部更新 安全 e.g. {"island_id":["program_id1","program_id2"]}
    
    
    feature_map: 
    
    
    
    
    
    
    

    def to_dict(self)->Dict[str,Any]:
        return self.model_dump()
    
    def from_dict(self,data:Dict[str,Any])->"GraphState":
        return GraphState(**data)
    
def init_graph_state(config:Config)->GraphState:
    '''
    初始化图的状态
    确保所有GraphState属性都被正确初始化
    '''
    # 验证必要的配置参数
    if config.init_program_path == "":
        raise ValueError("init_program is not set")
    if config.evalutor_file_path == "":
        raise ValueError("evaluator_file_path is not set")
    if config.island.num_islands <= 0:
        raise ValueError("num_islands must be greater than 0")
    
    # 提取文件信息
    file_extension = os.path.splitext(config.init_program_path)[1]
    if not file_extension:
        file_extension = ".py"  # 默认扩展名
    
    # 加载和处理初始程序
    code = load_initial_program(config.init_program_path)
    language = extract_code_language(code)
    
    # 生成唯一ID并评估初始程序
    id = str(uuid.uuid4())
    metrics = asyncio.run(direct_evaluate(config.evalutor_file_path, config.init_program_path, config))
    
    # 创建初始程序对象
    initial_program = Program(
        id=id,
        code=code,
        language=language,
        parent_id=None,
        generation=0,
        timestamp=time.time(),
        metrics=metrics,
        iteration_found=0,
    )
    
    # 初始化岛屿相关数据结构
    num_islands = config.island.num_islands
    islands_id = [str(uuid.uuid4()) for _ in range(num_islands)] # 岛屿的id 全局唯一 不会被更新 安全
    best_program_each_island = {island_id:id for island_id in islands_id} # 每个岛屿上最好的程序id 安全 e.g. {"island_id":["program_id1","program_id2"]}
    generation_count = {island_id:0 for island_id in islands_id} # 每一个岛屿当前的代数 安全 e.g. {"island_id":0}
    island_programs = {island_id:[id] for island_id in islands_id} # 各个岛屿上的程序id 安全 e.g. {"island_id":["program_id1","program_id2"]}
    all_programs = {id: initial_program}
    newest_programs = {island_id:id for island_id in islands_id} # 各个岛屿上最新的程序id 安全 e.g. {"island_id":"program_id"}
    status = {island_id:IslandStatus.INIT.value for island_id in islands_id} # 初始化每个岛屿的状态 安全 e.g. {"island_id":IslandStatus.INIT}
    archive = [id]  # 精英归档初始包含初始程序
    island_generation_count = {island_id:0 for island_id in islands_id} # 各个岛屿当前的代数 安全 e.g. {"island_id":0}
    island_evolution_direction = config.island.evolution_direction # 岛屿的进化方向 安全 e.g. {"island_id":"evolution_direction"}
    generation_count_in_meeting = 0 # 交流会进行的次数
    time_of_meeting = config.island.time_of_meeting # 每当各个岛屿迭代了time_of_meeting次 就会进行一次交流会 安全
    # 确保所有GraphState属性都被正确初始化
    return GraphState(
        # 全局共享状态
        init_program=id,
        best_program=code,
        best_program_id=id,
        best_program_each_island=best_program_each_island,
        best_metrics=metrics,
        generation_count=generation_count,
        num_islands=num_islands,
        archive=archive,
        island_programs=island_programs,
        all_programs=all_programs,
        evaluation_program=config.evalutor_file_path,
        newest_programs=newest_programs,
        language=language,
        file_extension=file_extension,
        status=status,
        island_generation_count=island_generation_count,
        islands_id=islands_id,
        # island_evolution_direction=island_evolution_direction,
        generation_count_in_meeting=generation_count_in_meeting,
        time_of_meeting=time_of_meeting
    )
    

    
    
    
    
if __name__ == "__main__":
    config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
    graph_state = init_graph_state(config)
    # print(graph_state.all_programs)
    all_programs = graph_state.all_programs
    print(len(all_programs))
    
    
        
    
    
    
    
    
    
    