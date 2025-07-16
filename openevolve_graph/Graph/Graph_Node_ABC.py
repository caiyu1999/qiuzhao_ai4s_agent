from openevolve_graph.program import Program
import random
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
from enum import Enum
from openevolve_graph.Config import Config
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ============================================================================
# LangGraph 兼容性说明
# ============================================================================
# 
# 这个抽象基类设计完全符合 LangGraph 的节点要求：
# 
# 1. 节点函数签名：每个节点类实现 __call__ 方法，接受 (state, config) 参数
# 2. 返回值格式：返回字典用于更新状态
# 3. 异步支持：AsyncNode 支持异步执行
# 4. 错误处理：统一的异常处理机制
# 
# 使用方式：
# ```python
# # 创建节点实例
# node = IslandEvolutionNode("island_0", config, island_id=0)
# 
# # 添加到 LangGraph
# builder.add_node("island_0", node)  # node 是可调用对象
# ```
# 
# ============================================================================

class NodeType(Enum):
    """节点类型枚举"""
    SYNC = "sync"           # 同步节点
    ASYNC = "async"         # 异步节点
    CONDITIONAL = "conditional"  # 条件节点
    PARALLEL = "parallel"   # 并行节点
    TERMINAL = "terminal"   # 终止节点

class NodeResult:
    """节点执行结果封装"""
    def __init__(self, 
                 state_updates: Optional[Dict[str, Any]] = None,
                 next_nodes: Optional[List[str]] = None,
                 error: Optional[Exception] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.state_updates = state_updates or {}
        self.next_nodes = next_nodes
        self.error = error
        self.metadata = metadata or {}
        
    def is_success(self) -> bool:
        return self.error is None

class BaseNode(ABC):
    """
    抽象基类：定义所有节点的通用接口
    
    这个基类确保所有节点都符合 LangGraph 的规范：
    - 实现 __call__ 方法，使实例可调用
    - 返回字典格式的状态更新
    - 统一的错误处理机制
    - 支持输入验证
    
    所有节点必须实现的基本方法：
    - execute: 执行节点逻辑
    - validate_input: 验证输入状态
    - get_node_type: 返回节点类型
    """
    
    def __init__(self, name: str, config: Optional[Config] = None):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def execute(self, state: BaseModel) -> Union[NodeResult, Dict[str, Any]]:
        """
        执行节点的核心逻辑
        
        Args:
            state: 当前图状态
            
        Returns:
            NodeResult 或 字典形式的状态更新
        """
        pass
    
    @abstractmethod
    def get_node_type(self) -> NodeType:
        """返回节点类型"""
        pass
    
    def validate_input(self, state: BaseModel) -> bool:
        """
        验证输入状态是否满足节点执行条件
        
        Args:
            state: 输入状态
            
        Returns:
            bool: 是否通过验证
        """
        return True
    
    def handle_error(self, error: Exception, state: BaseModel) -> NodeResult:
        """
        统一的错误处理
        
        Args:
            error: 发生的异常
            state: 当前状态
            
        Returns:
            NodeResult: 包含错误信息的结果
        """
        self.logger.error(f"Node {self.name} error: {error}")
        return NodeResult(error=error, metadata={"error_type": type(error).__name__})
    
    def __call__(self, state: BaseModel, config: Optional[Any] = None) -> Dict[str, Any]:
        """
        LangGraph 节点调用接口
        
        这个方法使类实例可以像函数一样被 LangGraph 调用
        符合 LangGraph 节点的标准签名：(state, config) -> dict
        
        Args:
            state: 当前图状态
            config: 可选的运行配置（LangGraph 传递的 RunnableConfig）
            
        Returns:
            Dict[str, Any]: 状态更新字典
        """
        try:
            # 验证输入
            if not self.validate_input(state):
                raise ValueError(f"Input validation failed for node {self.name}")
            
            self.logger.info(f"Executing node: {self.name}")
            
            # 执行节点逻辑
            result = self.execute(state)
            
            # 处理返回值，确保符合 LangGraph 期望的格式
            if isinstance(result, dict):
                # 直接返回字典（LangGraph 期望的格式）
                self.logger.info(f"Node {self.name} executed successfully")
                return result
            
            elif isinstance(result, NodeResult):
                # 转换 NodeResult 为字典
                if result.is_success():
                    self.logger.info(f"Node {self.name} executed successfully")
                    return result.state_updates
                else:
                    # 处理错误情况
                    if result.error:
                        raise result.error
                    else:
                        raise RuntimeError(f"Node {self.name} failed without specific error")
            else:
                raise ValueError(f"Invalid return type from node {self.name}: {type(result)}")
                
        except Exception as e:
            self.logger.error(f"Node {self.name} execution failed: {str(e)}")
            # 重新抛出异常，让 LangGraph 处理
            raise
    
    def get_node_info(self) -> Dict[str, Any]:
        """获取节点信息，用于调试和监控"""
        return {
            "name": self.name,
            "type": self.get_node_type().value,
            "class": self.__class__.__name__,
            "config": self.config.__dict__ if self.config else None
        }

class SyncNode(BaseNode):
    """同步节点基类"""
    def get_node_type(self) -> NodeType:
        return NodeType.SYNC

class AsyncNode(BaseNode):
    """异步节点基类"""
    
    def get_node_type(self) -> NodeType:
        return NodeType.ASYNC
    
    @abstractmethod
    async def execute_async(self, state: BaseModel) -> Union[NodeResult, Dict[str, Any]]:
        """异步执行方法"""
        pass
    
    def execute(self, state: BaseModel) -> Union[NodeResult, Dict[str, Any]]:
        """同步包装器，调用异步方法"""
        return asyncio.run(self.execute_async(state))

class ConditionalNode(BaseNode):
    """条件节点基类"""
    
    def get_node_type(self) -> NodeType:
        return NodeType.CONDITIONAL
    
    @abstractmethod
    def evaluate_condition(self, state: BaseModel) -> str:
        """
        评估条件，返回下一个节点的名称
        
        Args:
            state: 当前状态
            
        Returns:
            str: 下一个节点的名称
        """
        pass
    
    def execute(self, state: BaseModel) -> Union[NodeResult, Dict[str, Any]]:
        """条件节点只做路由，不修改状态"""
        next_node = self.evaluate_condition(state)
        return NodeResult(next_nodes=[next_node])

class ParallelNode(BaseNode):
    """并行节点基类"""
    
    def get_node_type(self) -> NodeType:
        return NodeType.PARALLEL
    
    @abstractmethod
    def get_parallel_tasks(self, state: BaseModel) -> List[str]:
        """
        获取需要并行执行的任务列表
        
        Args:
            state: 当前状态
            
        Returns:
            List[str]: 并行任务的节点名称列表
        """
        pass
    
    def execute(self, state: BaseModel) -> Union[NodeResult, Dict[str, Any]]:
        """并行节点返回需要并行执行的任务"""
        parallel_tasks = self.get_parallel_tasks(state)
        return NodeResult(next_nodes=parallel_tasks)



if __name__ == "__main__":
    from langgraph.graph import StateGraph, START, END
    config = Config.from_yaml("./openevolve_graph/test/test_config.yaml")
    # node_sync = SyncNode("sync_node",config)
    