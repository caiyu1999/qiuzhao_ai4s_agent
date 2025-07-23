''' 
这个文件实现一个graph_builder 根据config的文件生成graph
'''
from re import S
from openevolve_graph.Config import Config
from typing import List 
from langchain.chat_models import init_chat_model 
from dataclasses import dataclass 
from openevolve_graph.Graph import Graph_state
from langgraph.graph import StateGraph,START,END
from openevolve_graph.Graph.Graph_Node_ABC import BaseNode
from openevolve_graph.Graph.Graph_Node import *
from param import output
from openevolve_graph.models.structed_output import ResponseFormatter_template_diff,ResponseFormatter_template_rewrite# 这里为结构化输出的output模板
from openevolve_graph.Graph.Graph_edge import (
    routing_edge_llm_generate_failed,
    routing_edge_spinlock,
    routing_edge_evaluate_failed
    )

# --- 1. 定义日志目录和全局配置 ---
logger_dir = "/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/log"
config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")

# --- 2. 集中进行日志系统配置 ---
# 确保日志目录存在
if not os.path.exists(logger_dir):
    os.makedirs(logger_dir)

# 获取根记录器 (Root Logger)
log = logging.getLogger()
log.handlers.clear()  # 清空已有的handler
log.setLevel(logging.INFO) # 设置根记录器的级别

# 创建文件处理器，并设置更详细的格式
file_handler = logging.FileHandler(
    os.path.join(logger_dir, "test_run.log"), 
    mode='w', 
    encoding='utf-8'
)
# Formatter可以包含%(name)s来显示是哪个模块产生的日志
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

# 创建控制台处理器
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

# 为根记录器添加处理器
log.addHandler(file_handler)
log.addHandler(stream_handler)
# --- 日志配置结束 ---

class GraphBuilder:
    '''
    多个岛屿并行时候
    每一个岛屿的运行状态应该为一个subgraph 而不是superstep
    '''
    def __init__(self,config:Config):
        self.config = config
        self.num_islands = config.island.num_islands
        self.graph = None 
        self.is_build = False
        
    def build(self):
        node_init = node_init_status(config=self.config)
        main_graph_builder = StateGraph(Graph_state.GraphState)
        main_graph_builder.add_node("init",node_init)
        node_defer_end = node_defer(config=self.config)
        main_graph_builder.add_node("node_defer_end",node_defer_end,defer=True)
        for island_id in range(self.num_islands):
            subgraph = self.build_subgraph(island_id)
            main_graph_builder.add_node(f"subgraph_{island_id}",subgraph)
            main_graph_builder.add_edge(START,"init")
            main_graph_builder.add_edge("init",f"subgraph_{island_id}")
            main_graph_builder.add_edge(f"subgraph_{island_id}",f"node_defer_end")
        main_graph_builder.add_edge(f"node_defer_end",END)
        
        self.graph = main_graph_builder.compile()
        return self.graph
        
        
    def build_subgraph(self,island_id:int):
        '''
        构建一个岛屿的subgraph
        '''
        
        subgraph_builder = StateGraph(Graph_state.GraphState)
        config=self.config # 在child_evaluate节点后的routing节点
        
        
        node7 = node_update(config=config,island_id=str(island_id))
        node6 = node_spinlock(config=config,island_id=str(island_id))
        node5 = node_evaluate(config=config,island_id=str(island_id))
        node4 = node_llm_generate(config=config,island_id=str(island_id))
        node3 = node_build_prompt(config=config,island_id=str(island_id))
        node2 = node_sample_parent_inspiration(config=config,island_id=str(island_id))
        node1 = node_defer(config=config)
        node8 = node_defer(config=config)
        
        subgraph_builder.add_node(f"start_{island_id}",node1)
        subgraph_builder.add_node(f"sample_{island_id}",node2)
        subgraph_builder.add_node(f"build_prompt_{island_id}",node3)
        subgraph_builder.add_node(f"llm_generate_{island_id}",node4)
        subgraph_builder.add_node(f"child_evaluate_{island_id}",node5)
        subgraph_builder.add_node(f"spinlock_{island_id}",node6)
        subgraph_builder.add_node(f"update_{island_id}",node7)
        subgraph_builder.add_node(f"end_{island_id}",node8)
        
        subgraph_builder.add_edge(START,f"start_{island_id}")
        subgraph_builder.add_edge(f"start_{island_id}",f"sample_{island_id}")
        subgraph_builder.add_edge(f"sample_{island_id}",f"build_prompt_{island_id}")
        subgraph_builder.add_edge(f"build_prompt_{island_id}",f"llm_generate_{island_id}")
        # llm_generate节点后 需要判断是否失败 若失败则跳转到sample节点 成功则跳转到child_evaluate节点
        subgraph_builder.add_conditional_edges(f"llm_generate_{island_id}",
                                               routing_edge_llm_generate_failed(config=self.config,island_id=str(island_id)),
                                               {True:f"child_evaluate_{island_id}",False:f"sample_{island_id}"})
        
        # child_evaluate节点后 需要判断是否失败 若失败则跳转到sample节点 成功则跳转到update节点
        subgraph_builder.add_conditional_edges(f"child_evaluate_{island_id}",
                                               routing_edge_evaluate_failed(config=self.config,island_id=str(island_id)),
                                               {True:f"update_{island_id}",False:f"sample_{island_id}"})
        # 子图内部的更新不受外部影响
        # subgraph_builder.add_conditional_edges(f"spinlock_{island_id}",
        #                                        routing_edge_spinlock(config=self.config,island_id=str(island_id)),
        #                                        {True:f"update_{island_id}",False:f"spinlock_{island_id}"})
        subgraph_builder.add_edge(f"update_{island_id}",f"end_{island_id}")
        subgraph_builder.add_edge(f"end_{island_id}",END)
        
        _subgraph = subgraph_builder.compile()
        return _subgraph
    
    
    
    
    
if __name__ == "__main__":
    config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
    builder = GraphBuilder(config)
    # subgraph = builder.build_subgraph(0)
    graph = builder.build()
    # graph.get_graph().draw_mermaid_png(output_file_path='/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/Graph/graph.png')
    # subgraph.get_graph().draw_mermaid_png(output_file_path='/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/Graph/subgraph.png')
    # graph = builder.build()
    # # graph.get_graph().draw_mermaid_png(output_file_path='/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/Graph/graph.png')
    # # import asyncio
    
    # # # # import Ipython
    result = asyncio.run(graph.ainvoke(GraphState()))
    # import pdb;pdb.set_trace()

            
        
    
    
    