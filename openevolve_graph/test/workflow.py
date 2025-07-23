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

# config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
logger_dir = "/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/log"
logger = setup_logger(logger_dir, "test_run.log")


def build_main_graph(config:Config):
    node1 = node_init_status(config=config)
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("init_status",node1)
    graph_builder.add_edge(START,"init_status")
    graph_builder.add_edge("init_status",END)
    main_graph = graph_builder.compile()
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
    
    return subgraph_builder.compile()


@task
def run_island(island_graph , island_state:IslandState) -> dict:
    print(f"start run island {island_state.id}")
    result = island_graph.invoke(island_state)
    print(f"end run island {island_state.id}")
    return result 

@task
def run_main(graph_main,state:GraphState):
    result = graph_main.invoke(state)
    return result 

# @task 
# def run_meeting(config:Config,state:GraphState,island_state_list:List[IslandState]):
#     main_graph_state = meeting(config,state,island_state_list)
#     return main_graph_state

@entrypoint()
def main(dict_state:dict):
    state:GraphState = dict_state['state']
    config = dict_state['config']
    
    
    main_graph = build_main_graph(config)
    state_init = run_main(main_graph,state).result()
    state_init = GraphState.from_dict(state_init)
    num_islands = config.island.num_islands 
    
    
    island_state_dict = state_init.islands

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






