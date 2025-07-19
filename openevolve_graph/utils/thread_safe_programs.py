"""
线程安全程序容器模块

基于通用线程安全容器基类实现的程序特定容器
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from openevolve_graph.program import Program
import threading

@dataclass 
class ThreadSafePrograms:
    """
    线程安全的程序字典容器
    
    继承自 ThreadSafeContainer 基类，专门用于存储 Program 对象
    提供程序特定的便利方法
    """ 
    
    _programs: Dict[str, Program] = field(default_factory=dict)
    
    
    def _lock(self) -> threading.RLock:
        """获取锁"""
        return threading.RLock()
    
    def __contains__(self, key: str) -> bool:
        """检查是否包含键"""
        with self._lock():
            return key in self._programs
    def __len__(self) -> int:
        """获取程序数量"""
        with self._lock():
            return len(self._programs)

    def get(self,key:str)->Optional[Program | None | Any]:
        return self.get_program(key)
    # 实现基类的抽象方法
    def _internal_add(self, key: str, value: Program) -> None:
        """内部添加程序方法"""
        with self._lock():
            self._programs[key] = value
    
    def _internal_remove(self, key: str) -> Optional[Program]:
        """内部移除程序方法"""
        with self._lock():
            return self._programs.pop(key, None)
    
    def _internal_get(self, key: str) -> Optional[Program | None | Any]:
        """内部获取程序方法"""
        with self._lock():
            if key not in self._programs:
                raise ValueError(f"Program with ID {key} not found")
        with self._lock():
            return self._programs[key]
    
    def _internal_contains(self, key: str) -> bool:
        """内部包含检查方法"""
        with self._lock():
            return key in self._programs
    
    def _internal_keys(self) -> List[str]:
        """内部获取所有键方法"""
        with self._lock():
            return list(self._programs.keys())
    
    def _internal_values(self) -> List[Program]:
        """内部获取所有值方法"""
        with self._lock():
            return list(self._programs.values())
    
    def _internal_items(self) -> List[tuple[str, Program]]:
        """内部获取所有键值对方法"""
        with self._lock():
            return list(self._programs.items())
    
    def _internal_len(self) -> int:
        """内部获取长度方法"""
        with self._lock():
            return len(self._programs)
    
    def _internal_clear(self) -> None:
        """内部清空方法"""
        with self._lock():
            self._programs.clear()
    
    def _internal_update(self, items: Dict[str, Program]) -> None:
        """内部批量更新方法"""
        with self._lock():
            self._programs.update(items)
    
    def _internal_copy(self) -> "ThreadSafePrograms":
        """内部复制方法"""
        with self._lock():
            #将当前的程序字典复制 并返回新的线程安全程序容器
            return ThreadSafePrograms(self._programs.copy())
    
    # 程序特定的便利方法
    def add_program(self, program_id: str, program: Program) -> None:
        """添加程序（基类 add 方法的别名）"""
        with self._lock():
            self._internal_add(program_id, program)
    
    def remove_program(self, program_id: str) -> Optional[Program]:
        """移除程序（基类 remove 方法的别名）"""
        with self._lock():
            return self._internal_remove(program_id)
    
    def get_program(self, program_id: str) -> Optional[Program]:
        """获取程序（基类 get 方法的别名）"""
        with self._lock():
            return self._internal_get(program_id)
    
    def update_programs(self, programs: Dict[str, Program]) -> None:
        """批量更新程序（基类 update 方法的别名）"""
        with self._lock():
            self._internal_update(programs)
    def update_program(self, program_id: str, program: Program) -> None:
        """更新程序（基类 update 方法的别名）"""
        with self._lock():
            self._internal_update({program_id:program})
    
    def get_all_programs(self) -> Dict[str, Program | None | Any]:
        """获取所有程序（基类 copy 方法的别名） 并存放在一个字典中"""
        with self._lock():
            return {id:self._internal_get(id) for id in self._internal_keys()}
    def get_program_ids(self) -> List[str]:
        """获取所有程序ID（基类 keys 方法的别名）"""
        with self._lock():  
            return self._internal_keys()
        
    def values(self) -> List[Program]:
        """获取所有程序（基类 values 方法的别名）"""
        with self._lock():
            return self._internal_values()
    def copy(self) -> "ThreadSafePrograms":
        """复制程序容器"""
        with self._lock():
            return self._internal_copy()
    

    