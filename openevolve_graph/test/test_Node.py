from openevolve_graph.Graph.Graph_Node import *
from langgraph.graph import StateGraph, START, END
from openevolve_graph.Graph.Graph_state import GraphState
from openevolve_graph.Config.config import Config
from langgraph.types import Send
import os 
import pytest 
import logging
from langgraph.func import task,entrypoint

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


def test_Node_init_status(state,check=False):
    logging.info("--- 开始执行 test_Node_init_status ---")
    node = node_init_status(config=config)
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("init_status",node)
    graph_builder.add_edge(START,"init_status")
    graph_builder.add_edge("init_status",END)
    graph = graph_builder.compile()
    
    result = graph.invoke(state)
    
    # 添加类型断言，解决 linter 错误
    assert isinstance(result, dict), f"期望结果是字典，但得到了 {type(result)}"

    if check:
        logging.info(f"result from node_init_status: {result}")
        for k, v in result.items():
            logging.info(f"key__{k}:\n")
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    logging.info(f"{k_}:{v_}")
                    logging.info("-" * 100)
            else:
                logging.info(v)
            logging.info("=" * 100)
        
    return result 


def test_Node_sample(state, check=False):
    logging.info("--- 开始执行 test_Node_sample ---")
    num_islands = config.island.num_islands 
    
    node1 = node_init_status(config=config)
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("init_status",node1)
    graph_builder.add_edge(START,"init_status")
    graph_builder.add_edge("init_status",END)
    graph = graph_builder.compile()
    result = graph.invoke(state)
    
    
    
    
    
    
    def build_subgraph(island_id:str):
        subgraph_builder = StateGraph(IslandState)
        node_sample = node_sample_parent_inspiration(config=config,island_id=island_id)
        subgraph_builder.add_node("sample_parent_inspiration",node_sample)
        subgraph_builder.add_edge(START,"sample_parent_inspiration")
        subgraph_builder.add_edge("sample_parent_inspiration",END)
        return subgraph_builder.compile()
    
    subgraph_0 = build_subgraph("0")
    subgraph_1 = build_subgraph("1")
    subgraph_2 = build_subgraph("2")
    subgraph_3 = build_subgraph("3")
    
    init_state = graph.invoke(state)
    
    state_Graph = GraphState().from_dict(init_state)
    
    state_island_0 = subgraph_0.invoke(state_Graph.Island_0)
    state_island_1 = subgraph_1.invoke(state_Graph.Island_1)
    state_island_2 = subgraph_2.invoke(state_Graph.Island_2)
    state_island_3 = subgraph_3.invoke(state_Graph.Island_3)
    
    
    
    return state_island_0,state_island_1,state_island_2,state_island_3 

    
    
    
    # graph_builder.add_subgraph(subgraph_builder)
    
    
    
    
    
    
    # graph_builder.add_node("init_status",node1)
    # graph_builder.add_node("sample_parent_inspiration",node2)
    # graph_builder.add_edge(START,"init_status")
    # graph_builder.add_edge("init_status","sample_parent_inspiration")
    # graph_builder.add_edge("sample_parent_inspiration",END)
    # graph = graph_builder.compile()
    
    # result = graph.invoke(state)

    # # 添加类型断言，解决 linter 错误
    # assert isinstance(result, dict), f"期望结果是字典，但得到了 {type(result)}"

    # if check:
    #     logging.info(f"result from node_sample_parent_inspiration: {result}")
    #     for k, v in result.items():
    #         logging.info(f"key__{k}:\n")
    #         if isinstance(v, dict):
    #             for k_, v_ in v.items():
    #                 logging.info(f"{k_}:{v_}")
    #                 logging.info("-" * 100)
    #         else:
    #             logging.info(v)
    #         logging.info("=" * 100)
            
    # return result 

def test_Node_build_prompt(state, check=False):
    logging.info("--- 开始执行 test_Node_build_prompt ---")
    node3 = node_build_prompt(config=config,island_id="0")
    node2 = node_sample_parent_inspiration(config=config,island_id="0")
    node1 = node_init_status(config=config)
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("init_status",node1)
    graph_builder.add_node("sample_parent_inspiration",node2)
    graph_builder.add_node("build_prompt",node3)
    graph_builder.add_edge(START,"init_status")
    graph_builder.add_edge("init_status","sample_parent_inspiration")
    graph_builder.add_edge("sample_parent_inspiration","build_prompt")
    graph_builder.add_edge("build_prompt",END)
    graph = graph_builder.compile()
    result = graph.invoke(state)
    
    if check:
        logging.info(f"result from node_build_prompt: {result}")
        for k, v in result.items():
            logging.info(f"key__{k}:\n")
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    logging.info(f"{k_}:{v_}")
                    logging.info("-" * 100)
            else:
                logging.info(v)
            logging.info("=" * 100)
            
    return result 



def test_Node_llm_generate(state, check=False):
    logging.info("--- 开始执行 test_Node_llm_generate ---")
    node4 = node_llm_generate(config=config,island_id="0")
    node3 = node_build_prompt(config=config,island_id="0")
    node2 = node_sample_parent_inspiration(config=config,island_id="0")
    node1 = node_init_status(config=config)
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("init_status",node1)
    graph_builder.add_node("sample_parent_inspiration",node2)
    graph_builder.add_node("build_prompt",node3)
    graph_builder.add_node("llm_generate",node4)
    graph_builder.add_edge(START,"init_status")
    graph_builder.add_edge("init_status","sample_parent_inspiration")
    graph_builder.add_edge("sample_parent_inspiration","build_prompt")
    graph_builder.add_edge("build_prompt","llm_generate")
    graph_builder.add_edge("llm_generate",END)
    graph = graph_builder.compile()
    result = graph.invoke(state)
    if check:
        logging.info(f"result from node_build_prompt: {result}")
        for k, v in result.items():
            logging.info(f"key__{k}:\n")
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    logging.info(f"{k_}:{v_}")
                    logging.info("-" * 100)
            else:
                logging.info(v)
            logging.info("=" * 100)
            
    return result 


def test_Node_evaluate(state, check=False):
    logging.info("--- 开始执行 test_Node_llm_generate ---")
    node5 = node_evaluate(config=config,island_id="0")
    node4 = node_llm_generate(config=config,island_id="0")
    node3 = node_build_prompt(config=config,island_id="0")
    node2 = node_sample_parent_inspiration(config=config,island_id="0")
    node1 = node_init_status(config=config)
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("init_status",node1)
    graph_builder.add_node("sample_parent_inspiration",node2)
    graph_builder.add_node("build_prompt",node3)
    graph_builder.add_node("llm_generate",node4)
    graph_builder.add_node("evaluate",node5)
    graph_builder.add_edge(START,"init_status")
    graph_builder.add_edge("init_status","sample_parent_inspiration")
    graph_builder.add_edge("sample_parent_inspiration","build_prompt")
    graph_builder.add_edge("build_prompt","llm_generate")
    graph_builder.add_edge("llm_generate","evaluate")
    graph_builder.add_edge("evaluate",END)
    graph = graph_builder.compile()
    result = graph.invoke(state)
    if check:
        logging.info(f"result from node_build_prompt: {result}")
        for k, v in result.items():
            logging.info(f"key__{k}:\n")
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    logging.info(f"{k_}:{v_}")
                    logging.info("-" * 100)
            else:
                logging.info(v)
            logging.info("=" * 100)
            
    return result 

def test_Node_update(state, check=False):
    logging.info("--- 开始执行 test_Node_llm_generate ---")
    node7 = node_spinlock(config=config)
    node6 = node_update(config=config,island_id="0")
    node5 = node_evaluate(config=config,island_id="0")
    node4 = node_llm_generate(config=config,island_id="0")
    node3 = node_build_prompt(config=config,island_id="0")
    node2 = node_sample_parent_inspiration(config=config,island_id="0")
    node1 = node_init_status(config=config)
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("init_status",node1)
    graph_builder.add_node("sample_parent_inspiration",node2)
    graph_builder.add_node("build_prompt",node3)
    graph_builder.add_node("llm_generate",node4)
    graph_builder.add_node("evaluate",node5)
    graph_builder.add_node("update",node6)
    graph_builder.add_node("spinlock",node7)
    graph_builder.add_edge(START,"init_status")
    graph_builder.add_edge("init_status","sample_parent_inspiration")
    graph_builder.add_edge("sample_parent_inspiration","build_prompt")
    graph_builder.add_edge("build_prompt","llm_generate")
    graph_builder.add_edge("llm_generate","evaluate")
    graph_builder.add_edge("evaluate","spinlock")
    graph_builder.add_edge("spinlock","update")
    graph_builder.add_edge("update",END)
    graph = graph_builder.compile()
    result = graph.invoke(state)
    if check:
        logging.info(f"result from node_build_prompt: {result}")
        for k, v in result.items():
            logging.info(f"key__{k}:\n")
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    logging.info(f"{k_}:{v_}")
                    logging.info("-" * 100)
            else:
                logging.info(v)
            logging.info("=" * 100)
            
            
            
    return result 
if __name__ == "__main__":
    logging.info("主程序入口：开始测试...")
    state_origin = GraphState()
    # result = test_Node_init_status(state_origin,check=True)
    # result = test_Node_sample(state_origin,check=True)
    state_island_0,state_island_1,state_island_2,state_island_3 = test_Node_sample(state_origin,check=True)
    import pdb;pdb.set_trace()
    # graph.get_graph().draw_mermaid_png(output_file_path='/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/Graph/graph_sample.png')
    # result = graph.ainvoke(state_origin)
    # result = asyncio.run(result)
    # result = test_Node_build_prompt(state_origin,check=True)
    # result = test_Node_llm_generate(state_origin,check=True)
    # result = test_Node_evaluate(state_origin,check=True)
    # result = test_Node_update(state_origin,check=True)
    # import pdb;pdb.set_trace()
    
    
    
    
    
