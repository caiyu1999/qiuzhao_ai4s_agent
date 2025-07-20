from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from openevolve_graph.Graph.Graph_state import GraphState
from typing import Dict, Any
import time

def critical_node(state: GraphState) -> Dict[str, Any]:
    """
    关键节点 - 在执行期间需要独占状态访问
    """
    print("=== 进入关键节点 ===")
    print("此时其他节点被阻止执行")
    
    # 模拟关键操作
    time.sleep(2)
    
    print("=== 关键节点完成 ===")
    return {
        "status": ("critical", "completed"),
        "generation_count": {"0": state.generation_count.get("0", 0) + 1}
    }

def normal_node_1(state: GraphState) -> Dict[str, Any]:
    """普通节点1"""
    print("普通节点1执行")
    return {"status": ("node1", "completed")}

def normal_node_2(state: GraphState) -> Dict[str, Any]:
    """普通节点2"""
    print("普通节点2执行")
    return {"status": ("node2", "completed")}

# 方法1: 在关键节点前设置中断
def create_protected_graph():
    """创建带有保护的图"""
    builder = StateGraph(GraphState)
    
    # 添加节点
    builder.add_node("normal_1", normal_node_1)
    builder.add_node("critical", critical_node)
    builder.add_node("normal_2", normal_node_2)
    
    # 设置边
    builder.add_edge(START, "normal_1")
    builder.add_edge("normal_1", "critical")
    builder.add_edge("critical", "normal_2")
    builder.add_edge("normal_2", END)
    
    # 关键：在关键节点前设置中断，使用检查点
    memory = MemorySaver()
    graph = builder.compile(
        interrupt_before=["critical"],  # 在关键节点前中断
        checkpointer=memory  # 保存状态检查点
    )
    
    return graph

# 使用示例
if __name__ == "__main__":
    graph = create_protected_graph()
    
    # 第一阶段：执行到关键节点前
    thread_config = {"configurable": {"thread_id": "test_thread"}}
    initial_state = GraphState()
    
    print("阶段1: 执行到关键节点前")
    for event in graph.stream(initial_state, thread_config, stream_mode="values"):
        print(f"当前状态: {event}")
    
    # 检查状态 - 图在关键节点前停止
    state = graph.get_state(thread_config)
    print(f"下一个要执行的节点: {state.next}")
    
    # 第二阶段：继续执行关键节点（此时其他并发操作被阻止）
    print("\n阶段2: 执行关键节点")
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(f"当前状态: {event}") 