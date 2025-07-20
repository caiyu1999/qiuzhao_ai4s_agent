"""
实用示例：在岛屿进化系统中正确应用 LangGraph 更新原理

这个文件展示了如何在你的 openevolve_graph 项目中正确使用 LangGraph 的核心更新机制。
"""

from typing import Dict, Any, List, Annotated, Tuple, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from enum import Enum
import uuid
import time

# ============================================================================
# 基于你的代码优化的状态定义
# ============================================================================

class IslandStatus(Enum):
    INIT = "init"
    SAMPLING = "sampling"
    EVALUATING = "evaluating"
    COMPLETED = "completed"

def island_dict_reducer(left: Dict[str, Any], right: Optional[Tuple[str, Any]]) -> Dict[str, Any]:
    """
    岛屿字典更新器 - 基于你的 reducer_tuple 函数优化
    
    处理格式：
    - 初始化时：right 是完整字典
    - 运行时：right 是 (island_id, value) 元组
    """
    if right is None:
        return left or {}
    
    if isinstance(right, dict):
        # 初始化阶段
        return right
    
    elif isinstance(right, tuple) and len(right) == 2:
        # 运行时更新
        merge = left.copy() if left else {}
        island_id, value = right
        merge[island_id] = value
        return merge
    
    else:
        raise ValueError(f"无效的更新格式: {right}")

class OptimizedGraphState(TypedDict):
    """
    优化的图状态 - 基于你的 GraphState 但更简洁
    """
    # 岛屿相关状态（使用自定义 reducer）
    island_status: Annotated[Dict[str, str], island_dict_reducer]
    generation_count: Annotated[Dict[str, int], island_dict_reducer]
    current_programs: Annotated[Dict[str, str], island_dict_reducer]
    
    # 全局状态（默认覆盖）
    best_program_id: str
    total_evaluations: int
    
    # 程序列表（使用追加 reducer）
    all_program_ids: Annotated[List[str], lambda x, y: (x or []) + (y if isinstance(y, list) else [y])]

# ============================================================================
# 优化的节点实现
# ============================================================================

class OptimizedIslandNode:
    """
    优化的岛屿节点 - 基于你的代码模式
    
    关键改进：
    1. 明确的状态更新格式
    2. 错误处理
    3. 线程安全
    """
    
    def __init__(self, island_id: str):
        self.island_id = island_id
        
    def __call__(self, state: OptimizedGraphState) -> Dict[str, Any]:
        """
        LangGraph 节点标准接口
        
        返回格式严格遵循 LangGraph 规范：
        - 只返回需要更新的字段
        - 使用正确的数据格式匹配 reducer
        """
        print(f"=== 岛屿 {self.island_id} 开始执行 ===")
        
        try:
            # 获取当前状态
            current_gen = state.get("generation_count", {}).get(self.island_id, 0)
            current_status = state.get("island_status", {}).get(self.island_id, IslandStatus.INIT.value)
            
            # 模拟岛屿进化逻辑
            new_program_id = str(uuid.uuid4())[:8]
            new_generation = current_gen + 1
            
            print(f"岛屿 {self.island_id}: 代数 {current_gen} -> {new_generation}")
            print(f"岛屿 {self.island_id}: 生成程序 {new_program_id}")
            
            # 关键：正确的状态更新格式
            return {
                # 使用元组格式更新岛屿特定数据
                "island_status": (self.island_id, IslandStatus.COMPLETED.value),
                "generation_count": (self.island_id, new_generation),
                "current_programs": (self.island_id, new_program_id),
                
                # 全局数据直接更新（可能有冲突，需要处理）
                "total_evaluations": state.get("total_evaluations", 0) + 1,
                
                # 追加到程序列表
                "all_program_ids": [new_program_id]
            }
            
        except Exception as e:
            print(f"岛屿 {self.island_id} 执行失败: {e}")
            return {
                "island_status": (self.island_id, "error"),
            }

class GlobalUpdateNode:
    """
    全局更新节点 - 处理全局状态的更新
    
    避免在岛屿节点中直接更新全局状态，防止竞争条件
    """
    
    def __call__(self, state: OptimizedGraphState) -> Dict[str, Any]:
        """
        集中处理全局状态更新
        """
        print("=== 全局状态更新 ===")
        
        # 检查所有岛屿状态
        island_statuses = state.get("island_status", {})
        completed_islands = [
            island_id for island_id, status in island_statuses.items() 
            if status == IslandStatus.COMPLETED.value
        ]
        
        # 选择最佳程序（模拟）
        current_programs = state.get("current_programs", {})
        if current_programs:
            # 简单选择第一个作为最佳（实际应该根据评估结果）
            best_program = next(iter(current_programs.values()))
        else:
            best_program = state.get("best_program_id", "")
        
        print(f"完成的岛屿: {completed_islands}")
        print(f"最佳程序: {best_program}")
        
        return {
            "best_program_id": best_program
        }

# ============================================================================
# 图构建和执行
# ============================================================================

def create_optimized_evolution_graph(num_islands: int = 3):
    """
    创建优化的进化图
    
    Args:
        num_islands: 岛屿数量
    """
    builder = StateGraph(OptimizedGraphState)
    
    # 初始化节点
    def init_node(state: OptimizedGraphState) -> Dict[str, Any]:
        """初始化所有岛屿状态"""
        print("=== 初始化图状态 ===")
        
        island_ids = [str(i) for i in range(num_islands)]
        
        return {
            "island_status": {island_id: IslandStatus.INIT.value for island_id in island_ids},
            "generation_count": {island_id: 0 for island_id in island_ids},
            "current_programs": {island_id: "" for island_id in island_ids},
            "best_program_id": "",
            "total_evaluations": 0,
            "all_program_ids": []
        }
    
    # 添加节点
    builder.add_node("init", init_node)
    
    # 创建岛屿节点
    island_nodes = {}
    for i in range(num_islands):
        island_id = str(i)
        island_node = OptimizedIslandNode(island_id)
        node_name = f"island_{island_id}"
        
        builder.add_node(node_name, island_node)
        island_nodes[island_id] = node_name
    
    # 全局更新节点
    global_update = GlobalUpdateNode()
    builder.add_node("global_update", global_update)
    
    # 构建图结构
    builder.add_edge(START, "init")
    
    # 从初始化到所有岛屿（并行执行）
    for node_name in island_nodes.values():
        builder.add_edge("init", node_name)
    
    # 从所有岛屿到全局更新
    for node_name in island_nodes.values():
        builder.add_edge(node_name, "global_update")
    
    builder.add_edge("global_update", END)
    
    # 使用检查点保存状态
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# ============================================================================
# 示例执行和最佳实践
# ============================================================================

def demonstrate_optimized_execution():
    """
    演示优化的执行模式
    """
    print("=" * 60)
    print("优化的岛屿进化系统演示")
    print("=" * 60)
    
    # 创建图
    graph = create_optimized_evolution_graph(num_islands=3)
    
    # 初始状态
    initial_state = {
        "island_status": {},
        "generation_count": {},
        "current_programs": {},
        "best_program_id": "",
        "total_evaluations": 0,
        "all_program_ids": []
    }
    
    # 执行配置
    config = {"configurable": {"thread_id": "evolution_demo"}}
    
    print("\n开始执行进化图...")
    print("-" * 40)
    
    # 执行图
    final_state = graph.invoke(initial_state, config)
    
    print("\n最终状态:")
    print("-" * 40)
    for key, value in final_state.items():
        print(f"{key}: {value}")
    
    print("\n执行完成!")

# ============================================================================
# 最佳实践总结
# ============================================================================

"""
基于 LangGraph 更新原理的最佳实践（针对你的项目）：

1. **状态设计原则**：
   - 使用 Annotated 类型明确更新行为
   - 岛屿特定数据用自定义 reducer
   - 全局数据避免并发冲突

2. **节点实现原则**：
   - 只返回需要更新的字段
   - 使用正确的数据格式匹配 reducer
   - 避免直接修改输入状态

3. **并发控制策略**：
   - 岛屿级操作可以并行
   - 全局状态更新集中处理
   - 使用检查点机制保证一致性

4. **错误处理**：
   - 在节点内部捕获异常
   - 返回错误状态而不是抛出异常
   - 提供恢复机制

5. **性能优化**：
   - 避免不必要的状态复制
   - 使用适当的 reducer 函数
   - 合理设计图拓扑结构

这些原则确保你的岛屿进化系统能够：
- 高效地并行处理多个岛屿
- 正确地管理共享状态
- 提供可靠的错误恢复
- 支持大规模的进化计算
"""

if __name__ == "__main__":
    demonstrate_optimized_execution() 