from dataclasses import asdict, dataclass, field, fields
import time
from typing import Dict, Any, Optional#, List, Union, Tuple, Set, FrozenSet, Iterable, Iterator, Generator, Callable, TypeVar, Generic, AnyStr, AnyPath, AnyPathType, AnyPathTypeVar, AnyPathTypeVarTuple, AnyPathTypeVarTupleVar, AnyPathTypeVarTupleVarTuple, AnyPathTypeVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTuple, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVarTupleVar, AnyPathTypeVarTupleVarTupleVar




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