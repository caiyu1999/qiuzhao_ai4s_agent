import functools
import time
import uuid
from typing import Any, Callable, Dict, Optional
from openevolve_graph.utils.progress_monitor import progress_monitor
import logging

logger = logging.getLogger(__name__)

def monitor_node_progress(node_name: str, island_id: Optional[str] = None):
    """
    装饰器：监控节点执行进度
    
    Args:
        node_name: 节点名称
        island_id: 岛屿ID（可选）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 生成唯一的节点执行ID
            execution_id = f"{node_name}_{island_id}_{int(time.time() * 1000)}"
            
            try:
                # 开始监控节点执行
                progress_monitor.start_node(
                    node_id=execution_id,
                    node_name=node_name,
                    island_id=island_id,
                    message=f"开始执行 {node_name}"
                )
                
                # 执行原始函数
                result = func(*args, **kwargs)
                
                # 标记节点执行完成
                progress_monitor.complete_node(
                    node_id=execution_id,
                    message=f"{node_name} 执行完成",
                    details={"result_type": type(result).__name__}
                )
                
                return result
                
            except Exception as e:
                # 标记节点执行失败
                progress_monitor.fail_node(
                    node_id=execution_id,
                    error_message=f"{node_name} 执行失败: {str(e)}",
                    details={"error_type": type(e).__name__, "traceback": str(e)}
                )
                raise
                
        return wrapper
    return decorator

def monitor_async_node_progress(node_name: str, island_id: Optional[str] = None):
    """
    装饰器：监控异步节点执行进度
    
    Args:
        node_name: 节点名称
        island_id: 岛屿ID（可选）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # 生成唯一的节点执行ID
            execution_id = f"{node_name}_{island_id}_{int(time.time() * 1000)}"
            
            try:
                # 开始监控节点执行
                progress_monitor.start_node(
                    node_id=execution_id,
                    node_name=node_name,
                    island_id=island_id,
                    message=f"开始执行 {node_name}"
                )
                
                # 执行原始异步函数
                result = await func(*args, **kwargs)
                
                # 标记节点执行完成
                progress_monitor.complete_node(
                    node_id=execution_id,
                    message=f"{node_name} 执行完成",
                    details={"result_type": type(result).__name__}
                )
                
                return result
                
            except Exception as e:
                # 标记节点执行失败
                progress_monitor.fail_node(
                    node_id=execution_id,
                    error_message=f"{node_name} 执行失败: {str(e)}",
                    details={"error_type": type(e).__name__, "traceback": str(e)}
                )
                raise
                
        return async_wrapper
    return decorator

class NodeProgressContext:
    """
    节点进度上下文管理器，用于更细粒度的进度控制
    """
    def __init__(self, node_name: str, island_id: Optional[str] = None):
        self.node_name = node_name
        self.island_id = island_id
        self.execution_id = f"{node_name}_{island_id}_{int(time.time() * 1000)}"
        
    def __enter__(self):
        progress_monitor.start_node(
            node_id=self.execution_id,
            node_name=self.node_name,
            island_id=self.island_id,
            message=f"开始执行 {self.node_name}"
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            progress_monitor.complete_node(
                node_id=self.execution_id,
                message=f"{self.node_name} 执行完成"
            )
        else:
            progress_monitor.fail_node(
                node_id=self.execution_id,
                error_message=f"{self.node_name} 执行失败: {str(exc_val)}",
                details={"error_type": exc_type.__name__ if exc_type else "Unknown"}
            )
    
    def update_progress(self, percentage: float, message: str = "", details: Optional[Dict[str, Any]] = None):
        """更新节点执行进度"""
        progress_monitor.update_node_progress(
            node_id=self.execution_id,
            progress_percentage=percentage,
            message=message,
            details=details
        ) 