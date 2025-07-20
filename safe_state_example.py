from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from openevolve_graph.Graph.Graph_state import GraphState, IslandStatus
from openevolve_graph.Graph.Graph_Node import node_evaluate
from openevolve_graph.Config import Config
from typing import Dict, Any
import asyncio
import time
import threading

class ProtectedEvaluationNode:
    """
    受保护的评估节点 - 基于你现有的 node_evaluate 类
    在评估期间防止其他节点修改相关状态
    """
    
    def __init__(self, config: Config, island_id: str = None):
        self.config = config
        self.island_id = island_id
        self.evaluation_node = node_evaluate(config, island_id)
        self._evaluation_lock = threading.Lock()
        
    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """
        线程安全的评估节点调用
        """
        # 检查是否有其他评估正在进行
        if hasattr(state, 'evaluation_in_progress') and state.evaluation_in_progress:
            raise NodeInterrupt(f"岛屿 {self.island_id} 等待其他评估完成")
        
        try:
            # 标记评估开始
            result = {"evaluation_in_progress": True}
            
            if self.island_id is None:
                print("=== 开始初始程序评估（独占模式）===")
                # 初始评估 - 需要独占访问
                evaluation_result = self.evaluation_node(state)
                result.update(evaluation_result)
                print("=== 初始程序评估完成 ===")
            else:
                print(f"=== 开始岛屿 {self.island_id} 程序评估 ===")
                # 岛屿评估 - 并行安全
                evaluation_result = self.evaluation_node(state)
                result.update(evaluation_result)
                print(f"=== 岛屿 {self.island_id} 程序评估完成 ===")
            
            # 标记评估完成
            result["evaluation_in_progress"] = False
            return result
            
        except Exception as e:
            # 确保在异常情况下也解锁
            return {
                "evaluation_in_progress": False,
                "error": str(e)
            }

class ConditionalIslandNode:
    """
    条件性岛屿节点 - 只在允许的情况下执行
    """
    
    def __init__(self, config: Config, island_id: str, node_func):
        self.config = config
        self.island_id = island_id
        self.node_func = node_func
        
    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """
        检查条件后执行节点
        """
        # 检查是否有评估正在进行
        if getattr(state, 'evaluation_in_progress', False):
            # 如果有评估正在进行，这个节点等待
            raise NodeInterrupt(f"岛屿 {self.island_id} 节点等待评估完成")
        
        # 检查岛屿状态
        island_status = state.status.get(self.island_id, IslandStatus.INIT_STATE)
        if island_status in [IslandStatus.EVALUATE_CHILD, IslandStatus.INIT_EVALUATE]:
            # 如果岛屿正在评估，其他操作等待
            raise NodeInterrupt(f"岛屿 {self.island_id} 正在评估，其他操作等待")
        
        # 执行节点
        return self.node_func(state)

def create_controlled_execution_graph(config: Config):
    """
    创建具有控制执行流程的图
    """
    builder = StateGraph(GraphState)
    
    # 创建受保护的评估节点
    init_evaluate = ProtectedEvaluationNode(config, island_id=None)
    island_evaluate_0 = ProtectedEvaluationNode(config, island_id="0")
    island_evaluate_1 = ProtectedEvaluationNode(config, island_id="1")
    
    # 创建条件性节点（基于你的现有节点类）
    from openevolve_graph.Graph.Graph_Node import (
        node_init_status, node_sample_parent_inspiration, 
        node_build_prompt, node_llm_generate
    )
    
    init_node = node_init_status(config)
    sample_node_0 = ConditionalIslandNode(
        config, "0", 
        node_sample_parent_inspiration(config, "0", n=5)
    )
    sample_node_1 = ConditionalIslandNode(
        config, "1", 
        node_sample_parent_inspiration(config, "1", n=5)
    )
    
    # 添加节点
    builder.add_node("init_status", init_node)
    builder.add_node("init_evaluate", init_evaluate)
    builder.add_node("sample_0", sample_node_0)
    builder.add_node("sample_1", sample_node_1)
    builder.add_node("evaluate_0", island_evaluate_0)
    builder.add_node("evaluate_1", island_evaluate_1)
    
    # 设置边 - 确保评估节点的独占执行
    builder.add_edge(START, "init_status")
    builder.add_edge("init_status", "init_evaluate")  # 初始评估独占执行
    
    # 初始评估完成后，岛屿可以并行采样
    builder.add_edge("init_evaluate", "sample_0")
    builder.add_edge("init_evaluate", "sample_1")
    
    # 采样完成后进行岛屿评估（可能需要保护）
    builder.add_edge("sample_0", "evaluate_0")
    builder.add_edge("sample_1", "evaluate_1")
    
    builder.add_edge("evaluate_0", END)
    builder.add_edge("evaluate_1", END)
    
    # 编译时设置关键节点的中断
    memory = MemorySaver()
    graph = builder.compile(
        interrupt_before=["init_evaluate"],  # 在初始评估前中断
        checkpointer=memory
    )
    
    return graph

# 针对你的具体需求的使用方式
def safe_parallel_evolution():
    """
    安全的并行进化执行示例
    """
    # 加载配置（使用你的配置路径）
    config = Config.from_yaml("path/to/your/config.yaml")
    
    # 创建受控图
    graph = create_controlled_execution_graph(config)
    
    # 执行图
    initial_state = GraphState()
    thread_config = {"configurable": {"thread_id": "safe_evolution"}}
    
    print("开始安全并行进化...")
    
    try:
        # 第一阶段：到达初始评估前
        for event in graph.stream(initial_state, thread_config, stream_mode="values"):
            print(f"阶段1 - 状态: {event.get('status', 'unknown')}")
        
        # 检查图状态
        state = graph.get_state(thread_config)
        print(f"暂停在节点: {state.next}")
        
        # 继续执行初始评估（独占模式）
        print("\n开始初始评估（独占模式）...")
        for event in graph.stream(None, thread_config, stream_mode="values"):
            print(f"阶段2 - 状态: {event.get('status', 'unknown')}")
            
    except Exception as e:
        print(f"执行中断: {e}")
        # 处理中断，可能需要恢复执行

if __name__ == "__main__":
    # 演示概念
    print("LangGraph 状态保护机制演示")
    print("1. interrupt_before - 编译时设置中断点")
    print("2. NodeInterrupt - 运行时动态中断")  
    print("3. 条件节点 - 基于状态的执行控制")
    print("4. 线程锁 - 底层同步机制") 