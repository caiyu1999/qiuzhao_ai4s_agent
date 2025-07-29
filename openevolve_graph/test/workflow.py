<<<<<<< Updated upstream
from langgraph.func import task,entrypoint
from openevolve_graph.Graph.Graph_Node import * 
from langgraph.graph import END ,START ,StateGraph
from openevolve_graph.Graph.Graph_edge import routing_iteration_end, routing_llm_generate_successful,routing_evaluate_successful
import langgraph 
from langgraph.checkpoint.memory import MemorySaver
from openevolve_graph.Graph.meeting import meeting
from openevolve_graph.utils.log_setup import setup_logger
import logging
import os
=======
import os
import re
import sys
import json
import time
import tqdm
import random
import threading
import logging
import argparse

from langgraph.func import task, entrypoint
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver

from openevolve_graph.Config import Config
from openevolve_graph.get_logger import setup_root_logger

from openevolve_graph.Graph.Graph_Node import (
    node_init_status,
    node_sample_parent_inspiration,
    node_build_prompt,
    node_llm_generate,
    node_evaluate,
    node_update,
)

from openevolve_graph.Graph.Graph_state import (
    GraphState,
    IslandState,
)

from openevolve_graph.Graph.Graph_edge import (
    routing_iteration_end,
    routing_llm_generate_successful,
    routing_evaluate_successful,
)

from openevolve_graph.Graph.meeting import meeting

from openevolve_graph.visualization.socket_sc import SimpleServer
from openevolve_graph.visualization.vis import start_visualization

>>>>>>> Stashed changes

logger_dir = "/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/log"
<<<<<<< Updated upstream
logger = setup_logger(logger_dir, "test_run.log")
=======
logger = setup_root_logger(logger_dir, "INFO")
>>>>>>> Stashed changes




def build_main_graph(config: Config):
    node1 = node_init_status(config=config,next_meeting=10)
    graph_builder = StateGraph(GraphState)
<<<<<<< Updated upstream
    graph_builder.add_node("init_status",node1)
    graph_builder.add_edge(START,"init_status")
    graph_builder.add_edge("init_status",END)
    main_graph = graph_builder.compile()
=======
    graph_builder.add_node("init_status", node1)
    graph_builder.add_edge(START, "init_status")
    graph_builder.add_edge("init_status", END)
    main_graph = graph_builder.compile(checkpointer=InMemorySaver())
>>>>>>> Stashed changes
    return main_graph

def node_wait(state):
    return state

def build_subgraph(island_id: str, config: Config, draw_graph: bool = False):
    subgraph_builder = StateGraph(IslandState)

    node_sample = node_sample_parent_inspiration(config=config, island_id=island_id)
    node_prompt = node_build_prompt(config=config)
    node_llm = node_llm_generate(config=config)
    node_eva = node_evaluate(config=config)
    node_up = node_update(config=config)
<<<<<<< Updated upstream
    
    subgraph_builder.add_node("sample_parent_inspiration",node_sample)
    subgraph_builder.add_node("build_prompt",node_prompt)
    subgraph_builder.add_node("llm_generate",node_llm)
    subgraph_builder.add_node("evaluate",node_eva)
    subgraph_builder.add_node("update",node_up)
    subgraph_builder.add_node("wait",node_wait,defer=True)
    
    subgraph_builder.add_edge(START,"sample_parent_inspiration")
    subgraph_builder.add_edge("sample_parent_inspiration","build_prompt")
    subgraph_builder.add_edge("build_prompt","llm_generate")
    subgraph_builder.add_conditional_edges("llm_generate",
                                           routing_llm_generate_successful(config=config,island_id=island_id),
                                           {True:"evaluate",False:"sample_parent_inspiration"})
    subgraph_builder.add_conditional_edges("evaluate",
                                           routing_evaluate_successful(config=config,island_id=island_id),
                                           {True:"update",False:"sample_parent_inspiration"})
    subgraph_builder.add_conditional_edges("update",routing_iteration_end(config=config,island_id=island_id,meeting_interval=meeting_interval),
                                           {True:"wait",False:"sample_parent_inspiration"})
    subgraph_builder.add_edge("wait",END)
    
    return subgraph_builder.compile()
=======
>>>>>>> Stashed changes

    subgraph_builder.add_node("sample_parent_inspiration", node_sample)
    subgraph_builder.add_node("build_prompt", node_prompt)
    subgraph_builder.add_node("llm_generate", node_llm)
    subgraph_builder.add_node("evaluate", node_eva)
    subgraph_builder.add_node("update", node_up)
    subgraph_builder.add_node("wait", node_wait, defer=True)

    subgraph_builder.add_edge(START, "sample_parent_inspiration")
    subgraph_builder.add_edge("sample_parent_inspiration", "build_prompt")
    subgraph_builder.add_edge("build_prompt", "llm_generate")
    subgraph_builder.add_conditional_edges(
        "llm_generate",
        routing_llm_generate_successful(config=config, island_id=island_id),
        {True: "evaluate", False: "sample_parent_inspiration"}
    )
    subgraph_builder.add_conditional_edges(
        "evaluate",
        routing_evaluate_successful(config=config, island_id=island_id),
        {True: "update", False: "sample_parent_inspiration"}
    )
    subgraph_builder.add_conditional_edges(
        "update",
        routing_iteration_end(config=config, island_id=island_id),
        {True: "wait", False: "sample_parent_inspiration"}
    )
    subgraph_builder.add_edge("wait", END)
    subgraph = subgraph_builder.compile(checkpointer=InMemorySaver())
        
    if draw_graph:
        if config.graph_image_path != "":
            subgraph.get_graph().draw_mermaid_png(output_file_path=config.graph_image_path)
        else:
            image_path = os.path.dirname(config.init_program_path)
            image_path = os.path.join(image_path, f"subgraph.png")
            subgraph.get_graph().draw_mermaid_png(output_file_path=image_path)
    
    return subgraph

@task
<<<<<<< Updated upstream
def run_island(island_graph , island_state:IslandState) -> dict:
    print(f"start run island {island_state.id}")
    result = island_graph.invoke(island_state)
    print(f"end run island {island_state.id}")
    return result 

@task
def run_main(graph_main,state:GraphState):
    result = graph_main.invoke(state)
    return result 
=======
def run_island(island_graph, island_state: IslandState) -> dict:
    result = island_graph.invoke(island_state)
    return result

@task
def init_status(graph_main, state: GraphState):
    '''
    初始化节点状态
    '''
    result = graph_main.invoke(state)
    return result
>>>>>>> Stashed changes


def load_checkpoint(checkpoint_path: str,state: GraphState):
    '''
    这里的检查点是文件夹地址，如 ..../circle_packing/checkpoint/
    本函数用于从检查点路径中提取轮数,并新建一个GraphState对象
    
    '''
    # 检查检查点文件夹是否存在 如果不存在 返回原state并报错
    if not os.path.exists(checkpoint_path):
        logger.error(f"检查点文件夹 {checkpoint_path} 不存在")
        return state,0

    
    # 检查检查点文件夹是否为空 如果为空 返回原state并报错
    if not os.path.exists(checkpoint_path):
        logger.error(f"检查点文件夹 {checkpoint_path} 为空")
        return state,0

    state_path = os.path.join(checkpoint_path, "state.json")
    with open(state_path, "r") as f:
        state_dict = json.load(f)
    state_new = GraphState.model_validate(state_dict)
    
    return state_new,state_new.iteration


    
def save_checkpoint(config: Config, state: GraphState, generation_count_in_meeting: int):
    '''
    保存检查点 在初始程序所在文件夹中
    '''
    checkpoint_path = os.path.dirname(config.init_program_path)
    checkpoint_path = os.path.join(checkpoint_path, f"checkpoint_{generation_count_in_meeting}")
    
    os.makedirs(checkpoint_path, exist_ok=True)

    # 保存state
    json_name = f"state.json"
    json_path = os.path.join(checkpoint_path, json_name)
    with open(json_path, "w") as f:
        f.write(state.to_json())
    
    # 保存最佳程序
    if state.best_program:
        # 保存最佳程序的代码
        best_program_path = os.path.join(checkpoint_path, f"best_program{state.file_extension}")
        with open(best_program_path, "w") as f:
            f.write(state.best_program.code)
    
    # 保存最佳程序的信息（包括指标）
    best_program_info_path = os.path.join(checkpoint_path, "best_program_info.json")
    with open(best_program_info_path, "w") as f:
        json.dump(
                    {
                        "id": state.best_program.id,
                        "generation": state.best_program.generation,
                        "iteration": state.best_program.iteration_found,
                        "current_iteration": state.iteration,
                        "metrics": state.best_program.metrics,
                        "language": state.best_program.language,
                        "timestamp": state.best_program.timestamp,
                        "saved_at": time.time(),
                    },
                    f,
                    indent=2,
                )
    
    
    # 保存各个岛屿上的信息 
    island_state_dict = state.islands 
    for island_id, island_state in island_state_dict.items():
        island_state_path = os.path.join(checkpoint_path, f"island_{island_id}.json")
        with open(island_state_path, "w") as f:
            f.write(island_state.to_json())
            
    # 保存config信息
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, "w") as f:
        config.to_yaml(config_path)
    

@entrypoint()
<<<<<<< Updated upstream
def main(dict_state:dict):
    state:GraphState = dict_state['state']
=======
def main(dict_state: dict):
    # 读取初始化的state和config
    state: GraphState = dict_state['state']
>>>>>>> Stashed changes
    config = dict_state['config']
    
    
    main_graph = build_main_graph(config)
    state_init = run_main(main_graph,state).result()
    state_init = GraphState.from_dict(state_init)
    num_islands = config.island.num_islands 
    
    
    island_state_dict = state_init.islands

<<<<<<< Updated upstream
    island_state_list = []
    for k,v in island_state_dict.items():
        island_state_list.append(v)
    
    generation_count_in_meeting = 0 
    
    while generation_count_in_meeting < config.max_iterations:
        if config.random_meeting:
            meeting_interval = random.choice(config.meeting_interval_list)
        else:
            meeting_interval = config.meeting_interval
        
        logger.info(f"下一次meeting_interval: {meeting_interval}次迭代后进行")
        
        
        island_graph_list = [build_subgraph(str(i),config,meeting_interval) for i in range(num_islands)]
        futures = [run_island(subgraph_i,island_state_i) for subgraph_i,island_state_i in zip(island_graph_list,island_state_list)]
        results = [f.result() for f in futures]
        island_state_lists = [IslandState.from_dict(result) for result in results]
        
        main_graph_state = meeting(config,state_init,island_state_lists)
        
        island_state_lists = main_graph_state.islands
        
        generation_count_in_meeting += main_graph_state.iteration
        
    return main_graph_state


test_dict = {
    "state":GraphState(),
    "config":Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
}

import sys
result = main.invoke(test_dict, {"recursion_limit": sys.maxsize})


# print(result)

import pdb;pdb.set_trace()
=======
    # 在线程中启动服务器，避免阻塞
    server = SimpleServer(port=config.port)
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()
    time.sleep(2)  # 增加等待时间确保服务器完全启动
    
    
    # 构建主图 默认第一次开始(或者读取checkpoint后)10次迭代后进行meeting
    main_graph = build_main_graph(config)

    # 初始化节点状态
    state_init = init_status(main_graph, state)

    # 将状态转换为GraphState对象
    state_init = GraphState.from_dict(state_init.result())
    

    
    iterations = 0
    
    if config.checkpoint:
        checkpoint_path = config.checkpoint
        state_init,iterations = load_checkpoint(checkpoint_path,state_init)
    
    # 初始化可视化数据
    server.init_vis_data(state_init)
    
    # 启动可视化应用
    vis_app = start_visualization(config, server)
    vis_thread = threading.Thread(target=vis_app.run, daemon=True)
    vis_thread.start()
    time.sleep(1)  # 等待可视化应用启动


    if len(state_init.islands) != config.island.num_islands:
        raise ValueError(f"检查点中的岛屿数量与配置中的岛屿数量不一致,检查点中的岛屿数量为 {len(state_init.islands)},配置中的岛屿数量为 {config.island.num_islands}")
        

    while iterations < config.max_iterations:
        logger.info(f"-------------------------------迭代次数: {state_init.iteration}/{config.max_iterations}--------------------------------")
        num_islands = config.island.num_islands

        island_state_dict = state_init.islands
        island_state_list = [v for k, v in island_state_dict.items()]
        # print(island_state_list[0].now_meeting)

        island_graph_list = [build_subgraph(str(i), config) for i in range(num_islands)]

        futures = [run_island(subgraph_i, island_state_i) for subgraph_i, island_state_i in zip(island_graph_list, island_state_list)]

        results = [f.result() for f in futures]
        island_state_lists = [IslandState.from_dict(result) for result in results]

        # 获取state 并保存在本地 meeting_interval 是距离下次meeting的迭代次数
        main_graph_state,meeting_interval = meeting(config, state_init, island_state_lists)
        
        # 通过线程安全的方式更新可视化数据
        server.init_vis_data(main_graph_state)
        
        checkpoint_path = save_checkpoint(config, main_graph_state, meeting_interval)
        
        # 更新state_init
        state_init = main_graph_state

        # 更新迭代计数器
        iterations = main_graph_state.iteration
        
    # 当所有迭代运行完成 打印最好的程序和精度
    logger.info(f"最好的程序: {main_graph_state.best_program}, 精度: {main_graph_state.best_program.metrics}")

    return main_graph_state

>>>>>>> Stashed changes





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="OpenEvolve Graph 程序参数配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    parser.add_argument(
            "--config", 
            type=str,
            help="配置文件路径 (默认: config.yaml)",
            default="/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml"
        )
        
    parser.add_argument(
            "--iterations", 
            type=int, 
            help="最大迭代次数 (覆盖配置文件中的设置)",
            
        )
    
    parser.add_argument(
        "--init_program",
        type = str,
        help = ""
    )
    parser.add_argument(
        "--checkpoint",
        type = str,
        help = "检查点路径"
    )
    
    parser.add_argument(
        "--evaluate_program",
        type=str,
        help = "评估程序路径"
    )
    
    
    args = parser.parse_args()
    
    
    config = Config.from_yaml(args.config)
    
    config.init_program_path = args.init_program if args.init_program else config.init_program_path
    config.evalutor_file_path = args.evaluate_program if args.evaluate_program else config.evalutor_file_path
    
    config.max_iterations = args.iterations if args.iterations else config.max_iterations
    
    config.checkpoint = args.checkpoint if args.checkpoint else config.checkpoint
    config.resume = True

    test_dict = {
        "state": GraphState(),
        "config": config
    }


    result = main.invoke(test_dict, {"recursion_limit": sys.maxsize,
                                    "configurable": {"thread_id": "1"}})

