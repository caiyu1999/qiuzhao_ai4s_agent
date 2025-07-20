from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from openevolve_graph.Graph.Graph_state import GraphState
from typing import Dict, Any
import time

def critical_evaluation_node(state: GraphState) -> Dict[str, Any]:
    """
    关键评估节点 - 使用动态中断来确保独占访问
    """
    # 检查是否有其他关键操作正在进行
    if state.status.get("critical_operation", "") == "in_progress":
        raise NodeInterrupt("另一个关键操作正在进行，请等待完成")
    
    print("=== 开始关键评估 ===")
    
    # 标记关键操作开始
    result = {
        "status": ("critical_operation", "in_progress"),
        "generation_count": {"0": state.generation_count.get("0", 0) + 1}
    }
    
    # 模拟评估过程
    time.sleep(2)
    
    # 更新为完成状态
    result["status"] = ("critical_operation", "completed")
    
    print("=== 关键评估完成 ===")
    return result

def island_node(state: GraphState, island_id: str) -> Dict[str, Any]:
    """
    岛屿节点 - 在执行前检查关键操作状态
    """
    # 检查是否有关键操作正在进行
    if state.status.get("critical_operation", "") == "in_progress":
        raise NodeInterrupt(f"岛屿 {island_id} 等待关键操作完成")
    
    print(f"岛屿 {island_id} 正在执行")
    return {
        "status": (f"island_{island_id}", "completed"),
        "generation_count": {island_id: state.generation_count.get(island_id, 0) + 1}
    }

def create_dynamic_interrupt_graph():
    """创建使用动态中断的图"""
    builder = StateGraph(GraphState)
    
    # 添加节点
    builder.add_node("critical_eval", critical_evaluation_node)
    builder.add_node("island_0", lambda state: island_node(state, "0"))
    builder.add_node("island_1", lambda state: island_node(state, "1"))
    
    # 设置并行执行的边
    builder.add_edge(START, "critical_eval")
    builder.add_edge(START, "island_0")  # 并行执行
    builder.add_edge(START, "island_1")  # 并行执行
    
    builder.add_edge("critical_eval", END)
    builder.add_edge("island_0", END)
    builder.add_edge("island_1", END)
    
    # 使用检查点保存状态
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    return graph

# 使用示例
if __name__ == "__main__":
    graph = create_dynamic_interrupt_graph()
    initial_state = GraphState()
    config = {"configurable": {"thread_id": "dynamic_test"}}
    
    try:
        for event in graph.stream(initial_state, config, stream_mode="values"):
            print(f"状态更新: {event}")
    except Exception as e:
        print(f"中断捕获: {e}")
        
        # 继续执行
        print("恢复执行...")
        for event in graph.stream(None, config, stream_mode="values"):
            print(f"恢复后状态: {event}") 