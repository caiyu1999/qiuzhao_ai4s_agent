from openevolve_graph.Graph.Graph_state import GraphState
from openevolve_graph.Config import Config 
from openevolve_graph.Graph.Graph_state import IslandState
import time
import logging
# logger = logging.getLogger(__name__)


# 详细的日志类
class DetailedLogger:
    """详细的日志记录器，用于精准定位问题"""
    
    def __init__(self, name: str = "GraphNode"):
        self.name = name
        self.logger = logging.getLogger(__name__)
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[INFO] {timestamp} | {self.name} | {message}"
        if extra_info:
            log_msg += f" | {extra_info}"
        # 这里会对接上层的父代logger
        self.logger.info(log_msg)
    
    def error(self, message: str, **kwargs):
        """记录错误日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[ERROR] {timestamp} | {self.name} | {message}"
        if extra_info:
            log_msg += f" | {extra_info}"
        # 这里会对接上层的父代logger
        self.logger.error(log_msg)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[WARNING] {timestamp} | {self.name} | {message}"
        if extra_info:
            log_msg += f" | {extra_info}"
        # 这里会对接上层的父代logger
        self.logger.warning(log_msg)
    
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[DEBUG] {timestamp} | {self.name} | {message}"
        if extra_info:
            log_msg += f" | {extra_info}"
        # 这里会对接上层的父代logger
        self.logger.debug(log_msg)
    
    def step(self, step_name: str, **kwargs):
        """记录步骤日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[STEP] {timestamp} | {self.name} | {step_name}"
        if extra_info:
            log_msg += f" | {extra_info}"
        # 这里会对接上层的父代logger
        self.logger.info(log_msg)
logger = DetailedLogger("GraphEdge")


class routing_llm_generate_successful(object):
    '''
    
    '''
    def __init__(self,config:Config,island_id:str):
        self.config = config
        self.island_id = island_id
    def __call__(self,state:IslandState) -> bool:
        if state.llm_generate_success == True:#证明模型正常生成了答案 
            logger.info(f"Island:{state.id} llm_generate_success ")
            return True #跳转到下一个节点 
        else:
            logger.debug(f"Island:{state.id} llm_generate_success is False,turning to sample")
            return False #跳转到原节点

class routing_evaluate_successful(object):
    '''
    
    '''
    def __init__(self,config:Config,island_id:str):
        self.config = config
        self.island_id = island_id
    def __call__(self,state:IslandState) -> bool:
        if state.evaluate_success == True:#证明模型正常生成了答案 
            logger.info(f"Island:{state.id} evaluate_success ")
            return True #跳转到下一个节点 
        else:
            logger.debug(f"Island:{state.id} evaluate_success is False,turning to sample")
            return False #跳转到原节点

class routing_iteration_end(object):
    '''
    这个routing负责判定是否到达了迭代次数
    '''
    def __init__(self,config:Config,island_id:str):
        self.config = config
        self.island_id = island_id
        
    def __call__(self,state:IslandState) -> bool:
        
        if state.next_meeting <= 0:
            
            logger.info(f"Island:{state.id} has reached the time of meeting,turning to meeting")
            return True
        else:
            logger.info(f"Island:{state.id} now start iteration: {state.iteration},turning to sample")
            return False
        
        
