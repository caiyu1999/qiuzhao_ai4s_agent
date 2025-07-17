#!/usr/bin/env python3
"""
LangGraph并行性验证脚本
根据LangGraph官网推荐的验证方法
"""
import os
import sys
sys.path.append('/Users/caiyu/Desktop/langchain/openevolve_graph')

import time
import threading
from langgraph.graph import StateGraph, START, END
from openevolve_graph.Graph.Graph_state import init_graph_state
from openevolve_graph.Graph.Graph_Node import (
    node_sample_parent, node_sample_inspiration, 
    node_get_artifacts, get_top_programs
)
from openevolve_graph.Config import Config

class ParallelVerifier:
    """验证并行性的工具类"""
    
    def __init__(self):
        self.node_start_times = {}
        self.node_end_times = {}
        self.lock = threading.Lock()
        
    def log_start(self, node_name):
        with self.lock:
            self.node_start_times[node_name] = time.time()
            timestamp = time.strftime('%H:%M:%S.%f')[:-3]
            print(f"[{timestamp}] 🚀 节点 {node_name} 开始执行")
            
    def log_end(self, node_name):
        with self.lock:
            self.node_end_times[node_name] = time.time()
            duration = self.node_end_times[node_name] - self.node_start_times[node_name]
            timestamp = time.strftime('%H:%M:%S.%f')[:-3]
            print(f"[{timestamp}] ✅ 节点 {node_name} 执行完成 (耗时: {duration:.3f}s)")
            
    def analyze_parallelism(self):
        """分析并行性"""
        print("\n" + "="*60)
        print("📊 并行性分析报告")
        print("="*60)
        
        # 按开始时间排序
        sorted_starts = sorted(self.node_start_times.items(), key=lambda x: x[1])
        
        print("\n⏰ 节点执行时间线:")
        for node_name, start_time in sorted_starts:
            if node_name in self.node_end_times:
                end_time = self.node_end_times[node_name]
                duration = end_time - start_time
                print(f"  {node_name}: {start_time:.3f} -> {end_time:.3f} (时长: {duration:.3f}s)")
        
        # 分析并行性
        print("\n🔍 并行性检测:")
        base_time = min(self.node_start_times.values())
        
        parallel_groups = {}
        for node_name, start_time in self.node_start_times.items():
            relative_start = start_time - base_time
            # 如果在0.1秒内开始，认为是并行的
            time_group = round(relative_start, 1)
            if time_group not in parallel_groups:
                parallel_groups[time_group] = []
            parallel_groups[time_group].append(node_name)
        
        for time_group, nodes in parallel_groups.items():
            if len(nodes) > 1:
                print(f"  ⚡ 在 T+{time_group}s 时并行执行: {', '.join(nodes)}")
            else:
                print(f"  🔄 在 T+{time_group}s 时串行执行: {nodes[0]}")

class TimedNode:
    """带时间记录的节点包装器"""
    
    def __init__(self, original_node, node_name, verifier):
        self.original_node = original_node
        self.node_name = node_name
        self.verifier = verifier
        
    def __call__(self, state, config=None):
        self.verifier.log_start(self.node_name)
        # 模拟一些处理时间来更清楚地看到并行性
        time.sleep(0.1)
        result = self.original_node(state, config)
        self.verifier.log_end(self.node_name)
        return result

def main():
    """主函数"""
    print("🎯 LangGraph并行性验证工具")
    print("="*50)
    
    # 初始化
    config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
    graph_state = init_graph_state(config)
    verifier = ParallelVerifier()
    
    print(f"🏝️ 岛屿ID列表: {graph_state.islands_id}")
    
    def wait_node(state):
        verifier.log_start("wait_node")
        print(f"🔄 等待节点执行 - 所有岛屿状态: {state.get('status', 'N/A')}")
        time.sleep(0.1)
        verifier.log_end("wait_node")
        return state
    
    # 创建各个岛屿的节点实例并用TimedNode包装
    nodes = {}
    for i, island_id in enumerate(graph_state.islands_id):
        nodes[f'sample_parent_{i+1}'] = TimedNode(
            node_sample_parent(config=config, island_id=island_id),
            f'sample_parent_{i+1}',
            verifier
        )
        nodes[f'sample_inspiration_{i+1}'] = TimedNode(
            node_sample_inspiration(config=config, island_id=island_id, n=10),
            f'sample_inspiration_{i+1}',
            verifier
        )
        nodes[f'get_artifacts_{i+1}'] = TimedNode(
            node_get_artifacts(config=config, island_id=island_id),
            f'get_artifacts_{i+1}',
            verifier
        )
        nodes[f'get_top_programs_{i+1}'] = TimedNode(
            get_top_programs(config=config, island_id=island_id, n=10),
            f'get_top_programs_{i+1}',
            verifier
        )
    
    # 构建状态图
    builder = StateGraph(type(graph_state))
    
    # 添加所有节点
    for node_name, node in nodes.items():
        builder.add_node(node_name, node)
    
    builder.add_node("wait_node", wait_node)
    
    # 添加边连接
    print("\n🔗 设置图连接:")
    print("1. 并行启动 - 从START同时连接到4个采样父代节点")
    
    # 从START并行启动所有岛屿
    for i in range(4):
        builder.add_edge(START, f"sample_parent_{i+1}")
    
    print("2. 每个岛屿内部串行执行")
    for i in range(4):
        builder.add_edge(f"sample_parent_{i+1}", f"sample_inspiration_{i+1}")
        builder.add_edge(f"sample_inspiration_{i+1}", f"get_artifacts_{i+1}")
        builder.add_edge(f"get_artifacts_{i+1}", f"get_top_programs_{i+1}")
        builder.add_edge(f"get_top_programs_{i+1}", "wait_node")
    
    builder.add_edge("wait_node", END)
    
    # 编译图
    graph = builder.compile()
    
    print("\n🎯 开始执行并行性验证:")
    print("观察指标:")
    print("  ✅ 如果是并行: 多个 sample_parent 节点应该几乎同时开始")
    print("  ✅ 如果是串行: 节点会按顺序依次开始")
    print("  ✅ 岛屿内部: 每个岛屿内的节点应该串行执行")
    print("  ✅ 总时间: 并行执行应该比串行快很多")
    
    start_time = time.time()
    
    # 使用debug=True来观察执行步骤
    print("\n📋 开始图执行 (debug=True):")
    result = graph.invoke(graph_state, debug=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n⏱️ 总执行时间: {total_time:.3f}秒")
    print(f"🎯 最终状态: {result.get('status', 'N/A')}")
    
    # 分析并行性
    verifier.analyze_parallelism()
    
    # 性能分析
    print("\n📈 性能分析:")
    expected_serial_time = 0.1 * 16  # 16个节点 * 0.1秒每个
    parallel_efficiency = (expected_serial_time / total_time) * 100
    print(f"  📊 理论串行时间: {expected_serial_time:.1f}秒")
    print(f"  📊 实际执行时间: {total_time:.3f}秒")
    print(f"  📊 并行效率: {parallel_efficiency:.1f}%")
    
    if parallel_efficiency > 200:
        print("  ✅ 优秀！检测到明显的并行加速效果")
    elif parallel_efficiency > 120:
        print("  ✅ 良好！存在一定的并行性")
    else:
        print("  ⚠️ 可能是串行执行，请检查图结构")
    
    print("\n🔍 验证并行性的官方方法:")
    print("1. 观察上面的时间戳 - 并行节点应该有相近的开始时间")
    print("2. 查看debug输出中的'step'信息 - 并行节点会在同一step中执行")
    print("3. 比较总执行时间 - 并行执行应该明显更快")
    print("4. 检查'parallel_groups'分析 - 显示哪些节点真正并行执行")
    
    # 成功
    print("\n✅ 并行性验证完成！")

if __name__ == "__main__":
    main() 