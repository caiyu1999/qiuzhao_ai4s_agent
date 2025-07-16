''' 
这个文件实现一个graph_config类 与图相关的配置
'''

from dataclasses import dataclass 
from typing import Optional


@dataclass
class GraphConfig:
    """Configuration for the graph"""
    
    parallel_generate:bool = True 
    checkpoint_path:Optional[str] = None 
    
    



