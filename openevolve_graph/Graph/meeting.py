# from openevolve_graph.Graph.Graph_Node import *  
from openevolve_graph.Graph.Graph_state import GraphState,IslandState
from openevolve_graph.Config.config import Config 
from typing import List
from openevolve_graph.utils.utils import _is_better,safe_numeric_average
from openevolve_graph.utils.thread_safe_programs import Programs_container
from openevolve_graph.utils.utils import _calculate_feature_coords, _feature_coords_to_key
import logging 
logger = logging.getLogger(__name__)

def meeting(config:Config,
            main_graph_state:GraphState,
            island_states:List[IslandState]):
    '''
    meeting
    整合信息 并在各个岛屿之间进行迁移

    best_program : 四个岛屿之中最好的程序即可

    all_programs : 四个岛屿的程序库汇总即可

    archive: all_programs中表现在前{config.island.archive_size}个程序即可 

    feature_map: 计算all_programs中所有程序的featurecoord即可 

    ''' 

    # update best_program 
    candidate_programs = [island_state.best_program for island_state in island_states]
    if not candidate_programs:
        raise ValueError("没有候选程序")
    best_program = candidate_programs[0]
    for program in candidate_programs[1:]:
        if _is_better(program, best_program):
            best_program = program
    if best_program is not None:
        main_graph_state.best_program = best_program
    else:
        raise ValueError("best_program is None")
    logger.info(f"本次meeting选出的best_program id: {getattr(best_program, 'id', None)}，metrics: {getattr(best_program, 'metrics', {})}")

    # update all_programs 
    all_programs = {}
    # num_all_programs = 0
    for island_state in island_states:
        # 获取当前每个岛屿的所有程序
        island_programs = island_state.programs.get_all_programs() # {pid1:program1,pid2:program2,...}
        all_programs.update(island_programs)
        # num_all_programs += len(island_programs)

    # assert num_all_programs == len(all_programs),f"num_all_programs: {num_all_programs},len(all_programs): {len(all_programs)}"
    main_graph_state.all_programs = Programs_container.from_dict(all_programs)
    logger.info(f"本次meeting合并后all_programs总数: {len(all_programs)}")

    # update archive 
    # 获取所有程序的精度 并排序 取前{config.archive_size}个程序
    metrics_dict = {pid:program.metrics for pid,program in all_programs.items()}
    # 这里的排序是metrics分数越高越好
    sorted_metrics = sorted(metrics_dict.items(), key=lambda x: safe_numeric_average(x[1]), reverse=True)
    archive = {}
    for pid, _ in sorted_metrics[:config.archive_size]:
        archive[pid] = all_programs[pid]

    main_graph_state.archive = Programs_container.from_dict(archive)
    logger.info(f"本次meeting选出的archive数量: {len(archive)}，archive_size配置: {config.archive_size if hasattr(config, 'archive_size') else getattr(config.island, 'archive_size', None)}")

    # update feature_map 
    feature_map = {}
    for pid, program in all_programs.items():
        feature_coords = _calculate_feature_coords(config, main_graph_state, program)
        feature_key = _feature_coords_to_key(feature_coords)
        feature_map[feature_key] = pid
    main_graph_state.feature_map = feature_map
    logger.info(f"本次meeting生成的feature_map数量: {len(feature_map)}")

    # update island_states  
    for island_state in island_states:
        island_state.all_programs = main_graph_state.all_programs.copy()
        island_state.feature_map = main_graph_state.feature_map.copy()
        island_state.archive = main_graph_state.archive.copy()
        island_state.all_best_program = best_program

    main_graph_state.generation_count_in_meeting += 1 
    logger.info(f"meeting已进行次数: {main_graph_state.generation_count_in_meeting}")
    
    
    #update islands 
    for island_state in island_states:
        main_graph_state.islands[island_state.id] = island_state
        
        
    main_graph_state.iteration += main_graph_state.islands[island_state.id].iteration

    return main_graph_state

    
if __name__ == "__main__":
    a= {} 
    a.update({"a":1})
    a.update({"b":2})
    print(a)
        
        
        






