# from openevolve_graph.Graph.Graph_Node import *  
from openevolve_graph.Graph.Graph_state import GraphState,IslandState
from openevolve_graph.Config.config import Config 
from typing import List
from openevolve_graph.utils.utils import _is_better,safe_numeric_average
from openevolve_graph.utils.thread_safe_programs import Programs_container
from openevolve_graph.utils.utils import _calculate_feature_coords, _feature_coords_to_key
import logging 
import random
logger = logging.getLogger(__name__)

def meeting(config:Config,
            main_graph_state:GraphState,
            island_states:List[IslandState],
            ):


    # 生成meeting间隔 所有岛屿在到达间隔后会进行meeting和migration
    # migration 还没完成
    if config.random_meeting:
        meeting_interval = random.choice(config.meeting_interval_list)
    else:
        meeting_interval = config.meeting_interval

    # 更新最佳程序
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
    logger.info(f"本次meeting选出的best_program id: {getattr(best_program, 'id', None)},metrics: {getattr(best_program, 'metrics', {})}")


    # 更新所有程序
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


    # 更新精英归档
    # 获取所有程序的精度 并排序 取前{config.archive_size}个程序
    metrics_dict = {pid:program.metrics for pid,program in all_programs.items()}
    # 这里的排序是metrics分数越高越好
    sorted_metrics = sorted(metrics_dict.items(), key=lambda x: safe_numeric_average(x[1]), reverse=True)
    archive = {}
    for pid, _ in sorted_metrics[:config.archive_size]:
        archive[pid] = all_programs[pid]

    main_graph_state.archive = Programs_container.from_dict(archive)
    logger.info(f"本次meeting选出的archive数量: {len(archive)}，archive_size配置: {config.archive_size if hasattr(config, 'archive_size') else getattr(config.island, 'archive_size', None)}")

    # 更新特征坐标
    feature_map = {}
    for pid, program in all_programs.items():
        feature_coords = _calculate_feature_coords(config, main_graph_state, program)
        feature_key = _feature_coords_to_key(feature_coords)
        feature_map[feature_key] = pid
    main_graph_state.feature_map = feature_map
    logger.info(f"目前特征坐标数量: {len(feature_map)}")

    # 更新岛屿状态
    for island_state in island_states:
        island_state.all_programs = main_graph_state.all_programs.copy()
        island_state.feature_map = main_graph_state.feature_map.copy()
        island_state.archive = main_graph_state.archive.copy()
        island_state.all_best_program = best_program
        island_state.now_meeting = 0 
        island_state.next_meeting = meeting_interval
      

    main_graph_state.generation_count_in_meeting += 1 
    logger.info(f"meeting已进行次数: {main_graph_state.generation_count_in_meeting}")
    for island_state in island_states:
        main_graph_state.islands[island_state.id] = island_state

    main_graph_state.iteration = main_graph_state.islands['0'].iteration
    logger.info(f"meeting完成 距离下次meeting还有{meeting_interval}次迭代")
    return main_graph_state , meeting_interval

    
if __name__ == "__main__":
    a= {} 
    a.update({"a":1})
    a.update({"b":2})
    print(a)
        
        
        






