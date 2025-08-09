from colorama import init
from networkx import thresholded_random_geometric_graph
from workflow import load_checkpoint
from openevolve_graph.Graph.Graph_state import GraphState
from openevolve_graph.Graph.Graph_RAG import node_rag
from openevolve_graph.Config.config import Config
import argparse
import asyncio








if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="OpenEvolve Graph 程序参数配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    parser.add_argument(
            "--config", 
            type=str,
            help="配置文件路径 (默认: config.yaml)",
            default="/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/test_config.yaml"
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
    
    config.checkpoint = "/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/checkpoint_4"
    config.resume = True

    state_init =GraphState()
    state_new,_ = load_checkpoint(config.checkpoint,state_init)
    state_new.rag_doc_path = "/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/documents"
    state_new.vector_save_dir = "/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/vector_store"
    
    test_dict = {
        "state": state_init,
        "config": config
    }
    print(f"done stage I ")
    rag = node_rag(config)
    
    state_new = asyncio.run(rag.execute(state_new))
    # print(state_new)
    # print(rag.llm.invoke("hi").content)
    
    # from langchain.chat_models import init_chat_model 
    
    # llm = init_chat_model(model="gpt-4o-mini",temperature=0,api_key = 'sk-Maf9m5KxsypZQ76kF2qQ6lsqLs3PL0cm2Bs3XeOD1yl6Lk86',base_url="https://api.chatanywhere.tech")
    # print(llm.invoke("hi").content)