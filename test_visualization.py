#!/usr/bin/env python3
"""
测试可视化功能的简单脚本
"""

import time
import threading
from openevolve_graph.Config import Config
from openevolve_graph.Graph.Graph_state import GraphState, IslandState, IslandStatus
from openevolve_graph.visualization.socket_sc import SimpleServer
from openevolve_graph.visualization.vis import start_visualization

def create_test_state():
    """创建一个测试状态"""
    config = Config()
    config.island.num_islands = 4
    config.port = 8888
    
    # 创建测试状态
    state = GraphState()
    state.islands_id = ["0", "1", "2", "3"]
    state.num_islands = 4
    state.iteration = 0
    state.generation_count_in_meeting = 0
    
    # 创建岛屿状态
    for i in range(4):
        island_id = str(i)
        island_state = IslandState(id=island_id)
        island_state.iteration = i * 10
        island_state.status = IslandStatus.INIT_STATE
        island_state.next_meeting = 10
        island_state.now_meeting = i
        state.islands[island_id] = island_state
    
    return state, config

def main():
    print("🚀 启动可视化测试...")
    
    # 创建测试状态
    state, config = create_test_state()
    
    # 启动服务器
    print("📡 启动Socket服务器...")
    server = SimpleServer(port=config.port)
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()
    time.sleep(2)
    
    # 初始化可视化数据
    print("📊 初始化可视化数据...")
    server.init_vis_data(state)
    
    # 启动可视化应用
    print("🖥️ 启动可视化界面...")
    vis_app = start_visualization(config, server)
    vis_thread = threading.Thread(target=vis_app.run, daemon=True)
    vis_thread.start()
    time.sleep(2)
    
    print("✅ 可视化系统已启动，按 Ctrl+C 停止...")
    
    try:
        # 模拟数据更新
        for i in range(10):
            print(f"🔄 更新数据 {i+1}/10...")
            
            # 更新状态
            for island_id in state.islands:
                state.islands[island_id].iteration += 1
                state.islands[island_id].now_meeting += 1
            
            # 更新可视化数据
            server.init_vis_data(state)
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n⏹️ 停止可视化系统...")
        vis_app.stop()
        server.stop()

if __name__ == "__main__":
    main() 