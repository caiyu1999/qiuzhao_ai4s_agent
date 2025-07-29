#!/usr/bin/env python3
"""
测试logger配置的脚本
验证子logger是否能正确输出到控制台
"""

import sys
import os
sys.path.append('/Users/caiyu/Desktop/langchain/openevolve_graph')

from openevolve_graph.get_logger import setup_root_logger
from openevolve_graph.Graph.Graph_Node import node_init_status
from openevolve_graph.Config import Config
from openevolve_graph.Graph.Graph_state import GraphState

def test_logger():
    """测试logger配置"""
    print("=== 开始测试logger配置 ===")
    
    # 设置根logger
    logger_dir = "/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/log"
    logger = setup_root_logger(logger_dir, "INFO")
    
    print("✅ 根logger配置完成")
    
    # 测试根logger
    logger.info("这是来自根logger的信息")
    logger.warning("这是来自根logger的警告")
    
    # 测试子logger（模拟Graph_Node中的logger）
    import logging
    test_logger = logging.getLogger("openevolve_graph.Graph.Graph_Node")
    test_logger.info("这是来自Graph_Node子logger的信息")
    test_logger.warning("这是来自Graph_Node子logger的警告")
    
    # 测试其他子logger
    other_logger = logging.getLogger("openevolve_graph.test")
    other_logger.info("这是来自test子logger的信息")
    
    print("✅ 所有logger测试完成")
    print("=== 如果上面的日志信息都显示在控制台中，说明logger配置成功 ===")

def test_node_logger():
    """测试节点中的logger"""
    print("\n=== 开始测试节点logger ===")
    
    try:
        # 加载配置
        config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
        
        # 创建节点实例
        node = node_init_status(config=config)
        
        # 创建状态
        state = GraphState()
        
        print("✅ 节点创建成功，现在应该能看到节点执行过程中的日志输出")
        
        # 注意：这里不实际执行节点，因为需要异步执行
        # 只是验证logger配置是否正确
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_logger()
    test_node_logger() 