"""
线程安全程序容器模块

基于通用线程安全容器基类实现的程序特定容器
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from openevolve_graph.program import Program
import threading

@dataclass 
class Programs_container:
    """
    程序字典容器
    """ 
    
    _programs: Dict[str, Program] = field(default_factory=dict)

    
    
    
    def __contains__(self, key: str) -> bool:
        """检查是否包含键"""
        return key in self._programs
    def __len__(self) -> int:
        """获取程序数量"""
        return len(self._programs)

    def get(self,key:str)->Optional[Program | None | Any]:
        return self.get_program(key)
    # 实现基类的抽象方法
    def _internal_add(self, key: str, value: Program) -> None:
        """内部添加程序方法"""
        self._programs[key] = value
    
    def _internal_remove(self, key: str) -> Optional[Program]:
        """内部移除程序方法"""
        return self._programs.pop(key, None)
    
    def update_best_program(self,program:Program)->None:
        '''
        更新岛屿内部的最好程序
        '''
        
        #排序岛屿内部程序 依据metrics挑选出最佳程序 并更新best_program 
        best_program = sorted(self._programs.values(), key=lambda x: x.metrics["combined_score"], reverse=True)[0]
        self.best_program = best_program
        

        
    
    def _internal_get(self, key: str) -> Optional[Program | None | Any]:
        """内部获取程序方法"""
        if key not in self._programs:
            return None
        return self._programs[key]
    
    def _internal_contains(self, key: str) -> bool:
        """内部包含检查方法"""
        return key in self._programs
    
    def _internal_keys(self) -> List[str]:
        """内部获取所有键方法"""
        return list(self._programs.keys())
    
    def _internal_values(self) -> List[Program]:
        """内部获取所有值方法"""
        return list(self._programs.values())
    
    def _internal_items(self) -> List[tuple[str, Program]]:
        """内部获取所有键值对方法"""
        return list(self._programs.items())
    
    def _internal_len(self) -> int:
        """内部获取长度方法"""
        return len(self._programs)
    
    def _internal_clear(self) -> None:
        """内部清空方法"""
        self._programs.clear()
    
    def _internal_update(self, items: Dict[str, Program]) -> None:
        """内部批量更新方法"""
        self._programs.update(items)
    
    def _internal_copy(self) -> "Programs_container":
        """内部复制方法"""
        return Programs_container(self._programs.copy())
    
    # 程序特定的便利方法
    def add_program(self, program_id: str, program: Program) -> None:
        """添加程序（基类 add 方法的别名）"""
        self._internal_add(program_id, program)
    
    def remove_program(self, program_id: str) -> Optional[Program]:
        """移除程序（基类 remove 方法的别名）"""
        return self._internal_remove(program_id)
    
    def get_program(self, program_id: str) -> Optional[Program|None]:
        """获取程序（基类 get 方法的别名）"""
        return self._internal_get(program_id)
    
    def update_programs(self, programs: Dict[str, Program]) -> None:
        """批量更新程序（基类 update 方法的别名）"""
        
        self._internal_update(programs)
    def update_program(self, program_id: str, program: Program) -> None:
        """更新程序（基类 update 方法的别名）"""
        
        self._internal_update({program_id:program})
    
    def get_all_programs(self) -> Dict[str, Program | None | Any]:
        """获取所有程序（基类 copy 方法的别名） 并存放在一个字典中"""
        
        return {id:self._internal_get(id) for id in self._internal_keys()}
    
    def get_all_programs_to_list(self)->List[Program]:
        '''
        获取所有程序 并返回List[Program]
        '''
        return self._internal_values()
    
    def get_program_ids(self) -> List[str]:
        """获取所有程序ID（基类 keys 方法的别名）"""
          
        return self._internal_keys()
        
    def values(self) -> List[Program]:
        """获取所有程序（基类 values 方法的别名）"""
        
        return self._internal_values()
    def copy(self) -> "Programs_container":
        """复制程序容器"""
        
        return self._internal_copy()
    
    @classmethod
    def from_dict(cls,programs:Dict[str,Program])->"Programs_container":
        """从字典创建程序容器"""
        return cls(programs)
    
    
    
    def get_top_programs(self,num:int)->List[Program]:
        ''' 
        获取岛屿内部的顶级程序 并返回List[Program]
        '''
        # 获取岛屿内部的程序
        programs = self.get_all_programs_to_list() # List[Program]
        
        # 根据metrics进行排序 依据组合分数
        programs_sorted = sorted(programs, key=lambda x: x.metrics["combined_score"], reverse=True) # List[Program]
        
        return programs_sorted[:num]
    
    
    def get_num_programs(self)->int:
        '''
        获取岛屿内部的程序数量
        '''
        return len(self.get_all_programs_to_list())
    
    
    def add_programs(self,programs:List[Program])->None:
        '''
        添加程序
        '''
        for program in programs:
            self._internal_add(program.id,program)
            
            
    def remove_programs(self,programs:List[Program])->None:
        '''
        移除程序
        '''
        for program in programs:
            self._internal_remove(program.id)
            
    
    
    
    
    

            
            
        



    

    