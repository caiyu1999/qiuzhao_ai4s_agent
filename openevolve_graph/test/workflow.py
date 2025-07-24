from langgraph.func import task,entrypoint
from openevolve_graph.Graph.Graph_Node import * 
from langgraph.graph import END ,START ,StateGraph
from openevolve_graph.Graph.Graph_edge import routing_iteration_end, routing_llm_generate_successful,routing_evaluate_successful
import langgraph 
from langgraph.checkpoint.memory import MemorySaver
from openevolve_graph.Graph.meeting import meeting
# from openevolve_graph.utils.log_setup import setup_logger
from openevolve_graph.utils.progress_monitor import progress_monitor
from openevolve_graph.utils.node_monitor_decorator import NodeProgressContext
import logging
import os
from openevolve_graph.get_logger import get_logger
from langgraph.checkpoint.memory import InMemorySaver

# config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
logger_dir = "/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/log"
# logger = setup_logger(logger_dir, "test_run.log")
logger = get_logger(logger_dir, "test_run.log")
# 启动WebSocket服务器用于进度监控
progress_monitor.start_server_in_thread()
logger.info("进度监控WebSocket服务器已启动")


def build_main_graph(config:Config):
    node1 = node_init_status(config=config)
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("init_status",node1)
    graph_builder.add_edge(START,"init_status")
    graph_builder.add_edge("init_status",END)
    main_graph = graph_builder.compile(checkpointer=InMemorySaver())
    return main_graph

def node_wait(state):
    return state 
def build_subgraph(island_id:str,config:Config,meeting_interval:int):
    subgraph_builder = StateGraph(IslandState)
    
    node_sample = node_sample_parent_inspiration(config=config,island_id=island_id)
    node_prompt = node_build_prompt(config=config)
    node_llm = node_llm_generate(config=config)
    node_eva = node_evaluate(config=config)
    node_up = node_update(config=config)
    
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
    
    return subgraph_builder.compile(checkpointer = InMemorySaver())


@task
def run_island(island_graph , island_state:IslandState) -> dict:
    island_id = island_state.id
    with NodeProgressContext(f"Island_{island_id}_Execution", island_id) as progress:
        progress.update_progress(0, f"开始执行岛屿 {island_id}")
    
        
        progress.update_progress(50, f"岛屿 {island_id} 执行中...")
        result = island_graph.invoke(island_state)
        
        progress.update_progress(100, f"岛屿 {island_id} 执行完成")
    
    return result 

@task
def run_main(graph_main,state:GraphState):
    with NodeProgressContext("Main_Graph_Initialization") as progress:
        progress.update_progress(0, "开始初始化主图")
        result = graph_main.invoke(state)
        progress.update_progress(100, "主图初始化完成")
    return result 

# @task 
# def run_meeting(config:Config,state:GraphState,island_state_list:List[IslandState]):
#     main_graph_state = meeting(config,state,island_state_list)
#     return main_graph_state

@entrypoint()
def main(dict_state: dict):
    state: GraphState = dict_state['state']
    config = dict_state['config']

    with NodeProgressContext("Workflow_Main") as main_progress:
        main_progress.update_progress(5, "开始主工作流程")

        main_graph = build_main_graph(config)
        main_progress.update_progress(10, "主图构建完成")

        state_init = run_main(main_graph, state)
        state_init = GraphState.from_dict(state_init.result())
        num_islands = config.island.num_islands
        main_progress.update_progress(15, "状态初始化完成")

        island_state_dict = state_init.islands
        island_state_list = [v for k, v in island_state_dict.items()]

        generation_count_in_meeting = 0
        total_iterations = config.max_iterations

        main_progress.update_progress(20, f"开始迭代执行，总计 {total_iterations} 次迭代")

        while generation_count_in_meeting < config.max_iterations:
            iteration_progress = (generation_count_in_meeting / total_iterations) * 70 + 20  # 20-90%
            main_progress.update_progress(iteration_progress, f"迭代进度: {generation_count_in_meeting}/{total_iterations}")

            if config.random_meeting:
                meeting_interval = random.choice(config.meeting_interval_list)
            else:
                meeting_interval = config.meeting_interval

            logger.info(f"下一次meeting_interval: {meeting_interval}次迭代后进行")

            with NodeProgressContext("Island_Execution_Batch", f"batch_{generation_count_in_meeting}") as batch_progress:
                batch_progress.update_progress(0, f"开始第 {generation_count_in_meeting} 批次岛屿执行")

                island_graph_list = [build_subgraph(str(i), config, meeting_interval) for i in range(num_islands)]
                batch_progress.update_progress(25, "子图构建完成")

                futures = [run_island(subgraph_i, island_state_i) for subgraph_i, island_state_i in zip(island_graph_list, island_state_list)]
                batch_progress.update_progress(50, "岛屿执行任务已启动")

                results = [f.result() for f in futures]
                island_state_lists = [IslandState.from_dict(result) for result in results]
                batch_progress.update_progress(75, "岛屿执行完成，开始会议")

                main_graph_state = meeting(config, state_init, island_state_lists)
                batch_progress.update_progress(100, "会议完成")

            island_state_list = list(main_graph_state.islands.values())

            generation_count_in_meeting += main_graph_state.iteration

        main_progress.update_progress(100, "工作流程执行完成")

    return main_graph_state

test_dict = {
    "state":GraphState(),
    "config":Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
}

import sys
result = main.invoke(test_dict, {"recursion_limit": sys.maxsize,
                                 "configurable":{"thread_id":"1"}})


# print(result)

import pdb;pdb.set_trace()






