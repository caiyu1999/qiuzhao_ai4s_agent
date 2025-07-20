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
from openevolve_graph.Graph.Graph_edge import routing_edge_llm_generate_failed

class GraphBuilder:
    def __init__(self,config:Config):
        self.config = config
        self.num_islands = config.island.num_islands
        self.builder = StateGraph(Graph_state.GraphState)
        self.graph = None 
        self.is_build = False
        
        
    def add_node(self,node_name:str,node:BaseNode,defer:bool=False):
        self.builder.add_node(node_name,node,defer=defer)
    
    def add_edge(self,node_name:str,next_node_name:str):
        self.builder.add_edge(node_name,next_node_name)
        
    
    def build(self):
        self.add_node("init",node_init_status(config=self.config))
        self.add_node("init_evaluate",node_evaluate(config=self.config,island_id = None))
        self.add_node("defer",node_defer(config=self.config),defer=True)
        
        # add Node
        for island_id in range(self.num_islands):
            self.add_node(f"sample_{island_id}",node_sample_parent_inspiration(config=self.config,island_id=str(island_id)))
            self.add_node(f"build_prompt_{island_id}",node_build_prompt(config=self.config,island_id=str(island_id)))
            self.add_node(f"llm_generate_{island_id}",node_llm_generate(config=self.config,island_id=str(island_id)))
            self.add_node(f"generate_child_{island_id}",node_generate_child(config=self.config,island_id=str(island_id)))
            self.add_node(f"child_evaluate_{island_id}",node_evaluate(config=self.config,island_id=str(island_id)))
            
        # add Edge 
        for island_id in range(self.num_islands):
            self.add_edge("init_evaluate",f"sample_{island_id}")
            self.add_edge(f"sample_{island_id}",f"build_prompt_{island_id}")
            self.add_edge(f"build_prompt_{island_id}",f"llm_generate_{island_id}")
            self.builder.add_conditional_edges(f"llm_generate_{island_id}",routing_edge_llm_generate_failed(config=self.config,island_id=str(island_id)),{True:f"generate_child_{island_id}",False:f"sample_{island_id}"})
            self.add_edge(f"generate_child_{island_id}",f"child_evaluate_{island_id}")
            self.add_edge(f"child_evaluate_{island_id}","defer")
            
        self.builder.add_edge(START,"init")
        self.builder.add_edge("init",f"init_evaluate")
        self.builder.add_edge("defer",END)
        
        self.graph = self.builder.compile()
        
        self.is_build = True

        return self.graph
    
    
    
    
    
if __name__ == "__main__":
    config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
    builder = GraphBuilder(config)
    graph = builder.build()
    graph.get_graph().draw_mermaid_png(output_file_path='/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/Graph/graph.png')
    # import asyncio
    
    # # # import Ipython
    # result = asyncio.run(graph.ainvoke(GraphState()))
    # import pdb;pdb.set_trace()

            
        
    
    
    