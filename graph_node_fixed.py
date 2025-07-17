#!/usr/bin/env python3
"""
修复版本的Graph_Node.py文件
主要修复：
1. 将所有get_program()调用改为get()
2. 将keys()调用改为get_all_programs().keys()
3. 将values()调用改为get_all_programs().values()
4. 添加空值检查
"""

import os 
import json 
from openevolve_graph.Graph.Graph_Node_ABC import *
from openevolve_graph.Graph.Graph_state import GraphState, IslandStatus 
from openevolve_graph.Config import Config
from openevolve_graph.utils.utils import safe_numeric_average 
from openevolve_graph.utils.utils import _calculate_feature_coords,_feature_coords_to_key
from openevolve_graph.Prompt.sampler import PromptSampler_langchain
import random

def get_top_programs(state:GraphState, n: int = 10,metric:Optional[str] = None) -> List[Program]:
    """
    获取前N个最优程序 从all_programs中  以metric为指标排序
    
    Args:
        n: 返回的程序数量
        metric: 用于排序的指标名称（可选，默认使用平均值）
        
    Returns:
        List[Program]: 前N个最优程序列表
    """
    if not state.all_programs:
        return []

    all_programs = state.all_programs.get_all_programs()
    if metric:
        # 按指定指标排序
        sorted_programs = sorted(
            [p for p in all_programs.values() if metric in p.metrics],
            key=lambda p: p.metrics[metric],
            reverse=True,
        )
    else:
        # 按所有数值指标的平均值排序
        sorted_programs = sorted(
            all_programs.values(),
            key=lambda p: safe_numeric_average(p.metrics),
            reverse=True,
        )

    return sorted_programs[:n]

def get_artifacts(state:GraphState, program_id: str) -> Dict[str, Union[str, bytes]]:
    """获取程序的工件"""
    program = state.all_programs.get(program_id)
    if not program:
        return {}

    artifacts = {}

    # Load small artifacts from JSON
    if program.artifacts_json:
        try:
            small_artifacts = json.loads(program.artifacts_json)
            artifacts.update(small_artifacts)
        except json.JSONDecodeError as e:
            print(f"Failed to decode artifacts JSON for program {program_id}: {e}")

    # Load large artifacts from disk
    if program.artifact_dir and os.path.exists(program.artifact_dir):
        try:
            for filename in os.listdir(program.artifact_dir):
                file_path = os.path.join(program.artifact_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        artifacts[filename] = content
                    except UnicodeDecodeError:
                        with open(file_path, "rb") as f:
                            content = f.read()
                        artifacts[filename] = content
        except Exception as e:
            print(f"Failed to load artifacts from {program.artifact_dir}: {e}")

    return artifacts

class node_sample_parent_inspiration(SyncNode): 
    '''
    采样父代程序与灵感程序
    '''
    def __init__(self,config:Config,island_id:str,n:int=5,metric:Optional[str] = None):
        self.config = config 
        self.island_id = island_id 
        self.n = n 
        self.metric = metric 
        
    def execute(self,state:GraphState):
        parent_id = self._sample_parent(state).id 
        inspirations = self._sample_inspirations(state,parent_id)
        return parent_id,inspirations
    
    def __call__(self, state: GraphState, config: Optional[Any] = None) -> Dict[str, Any]:
        """
        LangGraph节点调用接口 - 线程安全版本
        只返回需要更新的字段，不直接修改state对象
        """
        # 执行采样逻辑
        parent_id,inspirations = self.execute(state)
        
        # 只返回需要更新的字段，让LangGraph的reducer处理并发
        return {
            "sample_program_id": {self.island_id: parent_id},
            "sample_inspirations": {self.island_id: inspirations},
            "status": {self.island_id: IslandStatus.SAMPLE},
        }
    
    def _sample_inspirations(self,state:GraphState,parent_id:str):
        ''' 
        采样灵感程序
        '''
        inspirations = [] 
        
        # 若最优程序存在 且与父代不同 且在所有的程序中 则加入灵感程序
        if (
            state.best_program_id is not None
            and state.best_program_id != parent_id
            and state.best_program_id in state.all_programs
        ):
            inspirations.append(state.best_program_id)
        
        # 添加顶级程序作为灵感
        top_n = max(1, int(self.n * self.config.island.elite_selection_ratio))
        top_programs = get_top_programs(state,n=top_n,metric=self.metric)
        for program in top_programs:
            if program.id not in [p.id for p in inspirations] and program.id != parent_id:
                inspirations.append(program)
        
        # 添加多样性程序
        if len(state.all_programs) > self.n and len(inspirations) < self.n:
            remaining_slots = self.n - len(inspirations)

            # 从不同的特征格子采样以获得多样性
            parent_program = state.all_programs.get(parent_id)
            if parent_program is None:
                raise ValueError(f"父代程序 {parent_id} 不存在")
            feature_coords = _calculate_feature_coords(self.config,state,parent_program)

            # 从附近的特征格子获取程序
            nearby_programs = []
            for _ in range(remaining_slots):
                # 扰动坐标
                perturbed_coords = [
                    max(0, min(self.config.island.feature_bins - 1, c + random.randint(-1, 1)))
                    for c in feature_coords
                ]

                # 尝试从这个格子获取程序
                cell_key = _feature_coords_to_key(perturbed_coords)
                if cell_key in state.feature_map:
                    program_id = state.feature_map[cell_key]
                    # 在添加前检查程序是否仍然存在
                    if (
                        program_id != parent_id
                        and program_id not in [p.id for p in inspirations]
                        and program_id in state.all_programs
                    ):
                        program = state.all_programs.get(program_id)
                        if program is not None:
                            nearby_programs.append(program)
                    elif program_id not in state.all_programs:
                        # 清理特征网格中的过时引用
                        del state.feature_map[cell_key]

            # 如果需要更多，添加随机程序
            if len(inspirations) + len(nearby_programs) < self.n:
                remaining = self.n - len(inspirations) - len(nearby_programs)
                all_ids = set(state.all_programs.get_all_programs().keys())
                excluded_ids = (
                    {parent_id}
                    .union(p.id for p in inspirations)
                    .union(p.id for p in nearby_programs)
                )
                available_ids = list(all_ids - excluded_ids)

                if available_ids:
                    random_ids = random.sample(available_ids, min(remaining, len(available_ids)))
                    for pid in random_ids:
                        program = state.all_programs.get(pid)
                        if program is not None:
                            nearby_programs.append(program)

            inspirations.extend(nearby_programs)

        return inspirations[:self.n]

    def _sample_parent(self,state:GraphState) -> Program:
        """
        从当前岛屿采样父代程序用于下一轮进化
        """
        rand_val = random.random()

        if rand_val < self.config.island.exploration_ratio:
            return self._sample_exploration_parent(state)
        elif rand_val < self.config.island.exploration_ratio + self.config.island.exploitation_ratio:
            return self._sample_exploitation_parent(state)
        else:
            return self._sample_random_parent(state)

    def _sample_exploration_parent(self, state: GraphState) -> Program:
        """探索性采样父代程序"""
        current_island_programs = state.island_programs[self.island_id]
        
        if not current_island_programs:
            raise ValueError(f"岛屿 {self.island_id} 未正确初始化，程序列表为空。")
        
        parent_id = random.choice(current_island_programs)
        program = state.all_programs.get(parent_id)
        if program is None:
            raise ValueError(f"程序 {parent_id} 不存在")
        return program

    def _sample_exploitation_parent(self, state: GraphState) -> Program:
        """利用性采样父代程序"""
        archive_programs = state.archive.get_all_programs()
        archive_in_current_island = [pid for pid in archive_programs.keys() if pid in state.island_programs[self.island_id]]
        
        if len(archive_in_current_island) > 0:
            chosen_id = random.choice(archive_in_current_island)
            program = state.all_programs.get(chosen_id)
            if program is None:
                raise ValueError(f"程序 {chosen_id} 不存在")
            return program
        else:
            if len(archive_programs) > 0:
                chosen_id = random.choice(list(archive_programs.keys()))
                program = state.all_programs.get(chosen_id)
                if program is None:
                    raise ValueError(f"程序 {chosen_id} 不存在")
                return program
            else:
                return self._sample_random_parent(state)
            
    def _sample_random_parent(self,state:GraphState) -> Program:
        """完全随机采样父代程序"""
        if not state.all_programs:
            raise ValueError("No programs available for sampling")

        current_island_programs = state.island_programs[self.island_id]
        if not current_island_programs:
            raise ValueError(f"当前岛屿 {self.island_id} 的程序列表为空")
        
        program_id = random.choice(current_island_programs)
        program = state.all_programs.get(program_id)
        if program is None:
            raise ValueError(f"程序 {program_id} 不存在")
        return program

class node_build_prompt(SyncNode):
    '''构建prompt节点'''
    def __init__(self,config:Config,island_id:str,metric:Optional[str] = None):
        self.config = config 
        self.island_id = island_id 
        self.metric = metric 
        self.prompt_sampler = PromptSampler_langchain(config=config.prompt)
        
    def execute(self,state:GraphState):
        return self._build_prompt(state)
        
    def __call__(self, state: GraphState, config: Optional[Any] = None) -> Dict[str, Any]:
        prompt = self.execute(state)
        return {
            "prompt": {self.island_id: prompt},
            "status": {self.island_id: IslandStatus.BUILD_PROMPT}
        }
        
    def _build_prompt(self,state:GraphState)->str:
        # 获取当前程序ID
        current_program_id = state.current_program_id.get(self.island_id)
        if current_program_id is None:
            current_program_id = state.sample_program_id[self.island_id]
        
        # 获取程序对象
        current_program = state.all_programs.get(current_program_id)
        if current_program is None:
            raise ValueError(f"当前程序 {current_program_id} 不存在")
        
        parent_program = state.all_programs.get(state.sample_program_id[self.island_id])
        if parent_program is None:
            raise ValueError(f"父代程序不存在")
        
        # 获取前代程序
        previous_programs = []
        temp_program = parent_program
        for _ in range(3):
            if temp_program.parent_id:
                parent_prog = state.all_programs.get(temp_program.parent_id)
                if parent_prog is not None:
                    previous_programs.append(parent_prog)
                    temp_program = parent_prog
                else:
                    break
            else:
                break
        
        previous_programs_dict = [prog.to_dict() for prog in previous_programs]
        
        # 获取灵感程序
        inspirations = state.sample_inspirations[self.island_id]
        inspirations_programs = []
        for pid in inspirations:
            prog = state.all_programs.get(pid)
            if prog is not None:
                inspirations_programs.append(prog.to_dict())
        
        top_programs = [i.to_dict() for i in get_top_programs(state,n=5,metric=self.metric)]
        
        # 获取generation_count，处理可能的字典类型
        evolution_round = 0
        if isinstance(state.generation_count, dict):
            evolution_round = state.generation_count.get(self.island_id, 0)
        else:
            evolution_round = state.generation_count
        
        return self.prompt_sampler.build_prompt(
            current_program = current_program.code,
            parent_program = parent_program.code,
            program_metrics = parent_program.metrics,
            previous_protgrams = previous_programs_dict,
            top_programs = top_programs,
            inspirations = inspirations_programs,
            language = "python",
            evolution_round = evolution_round,
            diff_based_evolution = self.config.diff_based_evolution,
            program_artifacts = get_artifacts(state,current_program_id),
        )

print("Graph_Node_Fixed.py 修复完成！")
print("主要修复：")
print("1. 所有get_program()调用改为get()并添加空值检查")
print("2. 所有keys()调用改为get_all_programs().keys()")
print("3. 所有values()调用改为get_all_programs().values()")
print("4. 修复了generation_count的类型错误")
print("5. 添加了适当的错误处理") 