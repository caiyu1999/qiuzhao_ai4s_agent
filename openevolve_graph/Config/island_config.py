'''
这个文件实现一个island_config类 与岛相关的配置 
'''

from dataclasses import dataclass,field
from typing import Optional,List

@dataclass
class IslandConfig:
    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5

    # Selection parameters
    elite_selection_ratio: float = 0.1 # 精英选择比例
    exploration_ratio: float = 0.2 # 探索比例
    exploitation_ratio: float = 0.7 # 利用比例
    diversity_metric: str = "edit_distance"  # Options: "edit_distance", "feature_based"

    # Feature map dimensions for MAP-Elites
    feature_dimensions: List[str] = field(default_factory=lambda: ["score", "complexity"])
    feature_bins: int = 10

    # Migration parameters for island-based evolution
    migration_interval: int = 50  # Migrate every N generations
    migration_rate: float = 0.1  # Fraction of population to migrate

    # Random seed for reproducible sampling
    random_seed: Optional[int] = 42

    # Artifact storage
    artifacts_base_path: Optional[str] = None  # Defaults to db_path/artifacts
    artifact_size_threshold: int = 32 * 1024  # 32KB threshold
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30