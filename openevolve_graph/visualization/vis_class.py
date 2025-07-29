import time
import random
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field,asdict
from threading import Thread, Lock
from openevolve_graph.Graph.Graph_state import (
    GraphState,
    IslandState,
    reducer_for_safe_container_all_programs,
    reducer_for_feature_map)
from openevolve_graph.program import Program 
from openevolve_graph.Config import Config
import logging 
logger=logging.getLogger(__name__)
'''
这个类主要实现岛屿可视化的进度更新的数据类别定义



'''

    
@dataclass
class IslandData_vis:
    '''
    岛屿可视化数据类
    '''
    id: str #岛屿id
    status: str = "" # 岛屿当前状态
    iterations: int = 0 # 岛屿总的迭代次数
    
    next_meeting: int = 0 # 岛屿距离下次会议时间
    now_meeting: int = 0 # 岛屿从上次会议到目前的迭代次数 进度条 = now_meeting/next_meeting 会议完成后自动清零并更新
    num_programs: int = 0 # 岛屿当前程序数量 
    latest_program_id: str = "" # 岛屿最新程序的id
    prompt: str = "" # 岛屿最新程序的prompt #这里只显示前50个字符即可 并且小一点
    sample_program_id: str = "" # 岛屿最新程序的父代程序id
    best_program_id: str = "" # 岛屿最好程序的id
    best_program_metrics: dict[str,float] = field(default_factory=dict) # 岛屿最好程序的精度
    def to_dict(self):
        return asdict(self)
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
    

@dataclass 
class best_program_vis:
    '''
    最好程序可视化数据类
    '''
    id: str = "" # 最好程序的id
    code: str = "" # 最好程序的代码
    from_island:str = "" #来自哪一个岛屿
    complexity: float =0.0 # 最好程序的复杂度
    diversity: float = 0.0 # 最好程序的多样性
    metrics: dict[str,float] = field(default_factory=dict) # 最好程序的指标
    sample_program_id: str = "" # 最好程序的父代程序id
    iteration_found:int =0 #发现该程序的迭代次数
    def to_dict(self):
        return asdict(self)
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
     

@dataclass 
class overall_information_vis:
    '''
    总体信息可视化数据类
    '''
    num_programs:int = 0 
    num_meetings:int = 0 # 会议次数
    def to_dict(self):
        return asdict(self)
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class visualize_data:
    '''
    可视化数据类 包含全部内容
    '''

    islands_data: dict[str,IslandData_vis] = field(default_factory=dict)
    best_program: best_program_vis = field(default_factory=best_program_vis)
    overall_information: overall_information_vis = field(default_factory=overall_information_vis)
    islands_state: dict[str,IslandState] = field(default_factory=dict)
    init_program: best_program_vis = field(default_factory=best_program_vis)

    def update_all(self,state: GraphState, init:bool = False):
        '''
        初始化可视化数据 在init或者checkpoint的GraphState会传入到这里 更新后会新建一个副本传给socket_server
        '''
        
        
        # 首先初始化各个岛屿的data
        for island_id in state.islands_id:
            island_state = state.islands[island_id]
            
            new_island_data = IslandData_vis(
            id = island_state.id,
            status = self._extract_status_value(island_state.status),
            iterations = island_state.iteration,
            next_meeting = island_state.next_meeting,
            now_meeting = island_state.now_meeting,
            num_programs = len(island_state.programs.get_all_programs()),
            latest_program_id = island_state.latest_program.id,
            prompt = island_state.prompt,
            sample_program_id = island_state.latest_program.parent_id if island_state.latest_program.parent_id else "",
            best_program_id = island_state.best_program.id,
            best_program_metrics = island_state.best_program.metrics,

        )
            self.islands_data[island_id] = new_island_data
        
        # 初始化best_program信息
        best_program = state.best_program
        logger.debug(f" best_program.id={best_program.id}, best_program.code exists={best_program.code is not None}, code length={len(best_program.code) if best_program.code else 0}, code content='{best_program.code[:50] if best_program.code else 'None'}...'")
        new_best_program_data = best_program_vis(
            id = best_program.id,
            from_island = best_program.island_id,
            metrics = best_program.metrics,
            complexity = best_program.complexity,
            diversity = best_program.diversity,
            iteration_found = best_program.iteration_found,
            code = best_program.code,
            sample_program_id = best_program.parent_id if best_program.parent_id else "",
        )
        self.best_program = new_best_program_data
        logger.debug(f" Created best_program_vis with code length: {len(new_best_program_data.code) if new_best_program_data.code else 0}, code meaningful={bool(new_best_program_data.code and new_best_program_data.code.strip())}")

        new_overall_information_data = overall_information_vis(
            num_programs = sum([len(island_state.programs.get_all_programs()) for island_state in state.islands.values()]),
            num_meetings = state.generation_count_in_meeting,
        )
        
        init_program = state.all_programs.get_program(state.init_program)
        self.init_program = best_program_vis(
            id = init_program.id,
            code = init_program.code,
            metrics=init_program.metrics,
            
        )
        self.overall_information = new_overall_information_data
        
        self.islands_state = state.islands
        
    def to_dict(self):
        return asdict(self)
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


    def update_after_meeting(self,state: GraphState):
        return self.update_all(state)
    
    
    def _extract_status_value(self, status_obj):
        """提取状态值，处理枚举对象"""
        if hasattr(status_obj, 'value'):
            return status_obj.value
        elif hasattr(status_obj, '_value_'):
            return status_obj._value_
        elif hasattr(status_obj, 'name'):
            return status_obj.name.lower()
        else:
            return str(status_obj)
    
    def update_after_node(self, node_info: dict):
        """根据节点信息更新可视化数据"""
        # 检查消息格式
        if "island_id" not in node_info or "update_dict" not in node_info:
            logger.warning(f"Warning: Invalid message format: {node_info}")
            return
            
        island_id = node_info["island_id"]
        update_dict = node_info["update_dict"]
        
        if island_id not in self.islands_data:
            logger.warning(f"Warning: Island {island_id} not found in visualization data")
            return
        
        island_state = self.islands_state[island_id]
        
        for key,value in update_dict.items():
            if key == "programs" and isinstance(value,tuple):
                island_state.programs = reducer_for_safe_container_all_programs(island_state.programs,value)
            
            elif key in ["status",
                         "iteration",
                         "next_meeting",
                         "now_meeting",
                         "prompt",
                         "sample_program",
                         "best_program",
                         "latest_program"]:
                # logger.debug(key,value)
                
                setattr(island_state, key, value)

        try:
            # logger.debug(type(island_state.programs))
            # logger.debug(island_state.now_meeting)
            new_island_data = IslandData_vis(
                id = island_state.id,
                status = self._extract_status_value(island_state.status),
                iterations = island_state.iteration,
                next_meeting = island_state.next_meeting,
                now_meeting = island_state.now_meeting,
                num_programs = len(island_state.programs.get_all_programs()),
                latest_program_id = island_state.latest_program.id,
                prompt = island_state.prompt,
                sample_program_id = island_state.latest_program.parent_id if island_state.latest_program.parent_id else "",
                best_program_id = island_state.best_program.id,
                best_program_metrics = island_state.best_program.metrics,
            )
            
                    
        except Exception as e:
            logger.error(f"Error updating island data: {e}")
            return
        
        
        
        self.islands_data[island_id] = new_island_data
        

        
        
        
        
    
    
    
    
