from dataclasses import asdict, dataclass, field, fields
import logging
import time 
from typing import Dict, Any, Optional#, List, Union, Tuple, Set, FrozenSet, Iterable, Iterator, Generator, Callable, TypeVar, Generic, AnyStr, AnyPath, AnyPathType, AnyPathTypeVar, AnyPathTypeVarTuple, AnyPathTypeVarTupleVar, AnyPathTypeVarTupleVarTuple, AnyPathTypeVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTuple, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVar
# from openevolve_graph.Graph.Graph_state import GraphState
# from openevolve_graph.Config import Config


logger = logging.getLogger(__name__)

@dataclass
class Program:
    """
    程序数据类，表示数据库中的一个程序
    
    该类使用@dataclass装饰器自动生成__init__、__repr__等方法
    包含程序的所有相关信息：标识、代码、进化信息、性能指标等
    """

    # 程序标识信息
    id: str  # 程序唯一标识符
    code: str  # 程序源代码
    language: str = "python"  # 编程语言，默认为Python

    # 进化相关信息
    parent_id: Optional[str] = None  # 父代程序ID，用于追踪进化谱系
    generation: int = 0  # 代数，表示程序在进化中的代数
    timestamp: float = field(default_factory=time.time)  # 创建时间戳
    iteration_found: int = 0  # 发现该程序的迭代次数

    # 性能指标
    metrics: Dict[str, float] = field(default_factory=dict)  # 程序性能指标字典

    # 衍生特征
    complexity: float = 0.0  # 程序复杂度
    diversity: float = 0.0  # 程序多样性

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外的元数据信息

    # 工件存储
    artifacts_json: Optional[str] = None  # JSON序列化的小型工件
    artifact_dir: Optional[str] = None  # 大型工件文件的路径

    def to_dict(self) -> Dict[str, Any]:
        """
        将程序对象转换为字典格式
        
        Returns:
            Dict[str, Any]: 程序对象的字典表示
        """
        return asdict(self)
    
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation"""
        # Get the valid field names for the Program dataclass
        valid_fields = {f.name for f in fields(cls)}

        # Filter the data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Log if we're filtering out any fields
        if len(filtered_data) != len(data):
            filtered_out = set(data.keys()) - set(filtered_data.keys())
            logger.debug(f"Filtered out unsupported fields when loading Program: {filtered_out}")

        return cls(**filtered_data)
    
if __name__ == "__main__":
    program = Program(id="1",code="print('Hello, World!')")
    print(program.to_dict().get("code",""))
    # print(program.get("id",""))

# async def _sample_exploration_parent(State:GraphState,config:Config) -> Program:
#     """
#     探索性采样父代程序（从当前岛屿）
    
#     Returns:
#         Program: 选中的父代程序
#     """
#     current_island_programs = State.island_programs[State.current_island]

#     if not current_island_programs:
#         # 如果当前岛屿为空，用最优程序或随机程序初始化
#         if self.best_program_id and self.best_program_id in self.programs:
#             # 将最优程序克隆到当前岛屿
#             best_program = self.programs[self.best_program_id]
#             self.islands[self.current_island].add(self.best_program_id)
#             best_program.metadata["island"] = self.current_island
#             logger.debug(f"Initialized empty island {self.current_island} with best program")
#             return best_program
#         else:
#             # 使用任何可用程序
#             return next(iter(self.programs.values()))

#     # 清理过时引用并从当前岛屿采样
#     valid_programs = [pid for pid in current_island_programs if pid in self.programs]

#     # 从岛屿中移除过时的程序ID
#     if len(valid_programs) < len(current_island_programs):
#         stale_ids = current_island_programs - set(valid_programs)
#         logger.debug(
#             f"Removing {len(stale_ids)} stale program IDs from island {self.current_island}"
#         )
#         for stale_id in stale_ids:
#             self.islands[self.current_island].discard(stale_id)

#     # 如果清理后没有有效程序，重新初始化岛屿
#     if not valid_programs:
#         logger.warning(
#             f"Island {self.current_island} has no valid programs after cleanup, reinitializing"
#         )
#         if self.best_program_id and self.best_program_id in self.programs:
#             best_program = self.programs[self.best_program_id]
#             self.islands[self.current_island].add(self.best_program_id)
#             best_program.metadata["island"] = self.current_island
#             return best_program
#         else:
#             return next(iter(self.programs.values()))

#     # 从有效程序中采样
#     parent_id = random.choice(valid_programs)
#     return self.programs[parent_id]

# def _sample_exploitation_parent(self) -> Program:
#     """
#     利用性采样父代程序（从归档/精英程序）
    
#     Returns:
#         Program: 选中的父代程序
#     """
#     if not self.archive:
#         # 如果没有归档，回退到探索采样
#         return self._sample_exploration_parent()

#     # 清理归档中的过时引用
#     valid_archive = [pid for pid in self.archive if pid in self.programs]

#     # 从归档中移除过时的程序ID
#     if len(valid_archive) < len(self.archive):
#         stale_ids = self.archive - set(valid_archive)
#         logger.debug(f"Removing {len(stale_ids)} stale program IDs from archive")
#         for stale_id in stale_ids:
#             self.archive.discard(stale_id)

#     # 如果没有有效的归档程序，回退到探索采样
#     if not valid_archive:
#         logger.warning(
#             "Archive has no valid programs after cleanup, falling back to exploration"
#         )
#         return self._sample_exploration_parent()

#     # 优先选择当前岛屿中的归档程序
#     archive_programs_in_island = [
#         pid
#         for pid in valid_archive
#         if self.programs[pid].metadata.get("island") == self.current_island
#     ]

#     if archive_programs_in_island:
#         parent_id = random.choice(archive_programs_in_island)
#         return self.programs[parent_id]
#     else:
#         # 如果当前岛屿没有归档程序，回退到任何有效的归档程序
#         parent_id = random.choice(valid_archive)
#         return self.programs[parent_id]

# def _sample_random_parent(self) -> Program:
#     """
#     完全随机采样父代程序
    
#     Returns:
#         Program: 选中的父代程序
#     """
#     if not self.programs:
#         raise ValueError("No programs available for sampling")

#     # 从所有程序中随机采样
#     program_id = random.choice(list(self.programs.keys()))
#     return self.programs[program_id]