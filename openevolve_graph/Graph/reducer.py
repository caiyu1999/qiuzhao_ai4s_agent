from typing import Any,Dict,List,Optional,Tuple
import logging
logger = logging.getLogger(__name__)

from openevolve_graph.utils.thread_safe_programs import Programs_container
from openevolve_graph.program import Program
from utils.utils import _is_better





def reducer_for_single_parameter(left:Any,right:Any)->Any:
    '''
    这个函数用来更新单个参数  这个参数在迭代过程中是永远不变的 
    '''
    return right 

def reducer_tuple(left: Dict[str, Any], right: Optional[Dict[str,Any]]|Tuple[str,Any]|List[Tuple[str,Any]]|None):
    '''
    在初始化时 传入的right和left是一个类型  用right替换left即可
    在运行过程中 传入的right是tuple类型 需要将tuple中的值更新到left中
    '''
    #print(f"reducer_tuple left: {left}, right: {right}")
    if right is None:
        return left 
    
    if isinstance(right,dict):
        #print(f"reducer_tuple right is a dict")
        return right
    
    elif isinstance(right,list):
        merge = left.copy()
        for item in right:
            if len(item) == 2:
                island_id = item[0]
                value = item[1]
                merge[island_id] = value
            else:
                raise ValueError("reducer_tuple right must be a tuple of length 2, but you give me a {}".format(len(item)))
    elif isinstance(right,tuple):
        merge = left.copy()
        if len(right) == 2:
            island_id = right[0]
            value = right[1]
            merge[island_id] = value
        else:       
            raise ValueError("reducer_tuple right must be a tuple of length 2 or 3, but you give me a {}".format(len(right)))
    else:
        logger.error(f"reducer_tuple right must be a dict, list, tuple, but you give me a {type(right)}")
    return merge 

def reducer_for_feature_map(left:Dict[str,Any],right:Optional[Tuple[str,str]]|Dict[str,Any]|Tuple[str,str,str])->Dict[str,Any]:
    if right is None:
        return left 
    
    if isinstance(right,dict):
        return right 
    
    elif isinstance(right,tuple):
        if len(right) == 2:
            feature_key = right[0]
            program_id = right[1]
            left[feature_key] = program_id
        else:
            raise ValueError("reducer_for_feature_map right must be a tuple of length 2, but you give me a {}".format(len(right)))
    
# def reducer_for_safe_container_island_programs(left:Dict[str,ThreadSafePrograms],right:Optional[Tuple[str,str,Program]|Tuple[str,str,str,Program]| Dict[str,ThreadSafePrograms]]|List[Tuple[str,Any]])->Dict[str,ThreadSafePrograms]:
#     '''
#     这里需要定义新的程序添加时的更新方式 
#     '''
#     #print(f"reducer_for_safe_container_island_programs left: {left}, right: {right}")
#     #========传入为None的情况========
#     if right is None:
#         return left 
    
#     #========传入为dict的情况========
#     if isinstance(right,dict):
#         return right
    
#     #========传入为list的情况========
#     elif isinstance(right,list): #添加，删除，更新多个程序 或者 替换多个程序
#         merge = left.copy()
#         for item in right:
#             #========传入为list[tuple] len(tuple) == 3 的情况========
#             if len(item) == 3:
#                 operation = item[0]
#                 island_id = item[1]
#                 program = item[2]
#                 if operation == "add":
#                     merge[island_id].add_program(program.id,program)
#                 elif operation == "remove":
#                     merge[island_id].remove_program(program.id)
#                 elif operation == "update":
#                     merge[island_id].update_program(program.id,program)
#                 else:
#                     raise ValueError("reducer_for_safe_container_island_programs right must be a tuple of length 3, but you give me a {}".format(len(item)))
#             #========传入为list[tuple] len(tuple) == 5 的情况========
#             if len(item) == 4:
#                 operation = item[0]
#                 island_id = item[1]
#                 program_need_replace_id = item[2]# 需要被替换的程序id
#                 program_replace_with = item[3]# 替换的程序id
                
#                 if operation == 'replace':
#                     merge[island_id].remove_program(program_need_replace_id)
#                     merge[island_id].add_program(program_replace_with.id,program_replace_with)
#                 else:
#                     raise ValueError("reducer_for_safe_container_island_programs right must be a tuple of length 4, but you give me a {}".format(len(item)))
#         return merge 
    
    
#     #========传入为tuple的情况========
#     elif isinstance(right,tuple): #添加，删除，更新单个程序 或者 替换单个程序
#         #========传入为tuple len(tuple) == 3 的情况========
#         if len(right) == 3:
#             merge = left.copy()
#             operation = right[0]
#             island_id = right[1]
#             program = right[2]
#             if operation == "add":
#                 merge[island_id].add_program(program.id,program)    
#             elif operation == "remove":
#                 merge[island_id].remove_program(program.id)
#             elif operation == "update":
#                 merge[island_id].update_program(program.id,program)
#             else:
#                 raise ValueError("reducer_for_safe_container_island_programs right must be a tuple of length 3, but you give me a {}".format(len(right)))
#             return merge 
#         #========传入为tuple len(tuple) == 5 的情况========
#         if len(right) == 4:
#             merge = left.copy()
#             operation = right[0]
#             island_id = right[1]
#             program_need_replace_id = right[2]# 需要被替换的程序id
#             program_replace_with = right[3]# 替换的程序id
#             if operation == 'replace':
#                 merge[island_id].remove_program(program_need_replace_id)
#                 merge[island_id].add_program(program_replace_with.id,program_replace_with)
#             else:
#                 raise ValueError("reducer_for_safe_container_island_programs right must be a tuple of length 4, but you give me a {}".format(len(right)))
    
def reducer_for_safe_container_all_programs(left:ThreadSafePrograms,right:Optional[Tuple[str,Program]|ThreadSafePrograms]|List[Tuple[str,Any]]|Tuple[str,str,Program])->ThreadSafePrograms:
    '''
    这里需要定义新的程序添加时的更新方式 
    '''
    #print(f"reducer_for_safe_container_all_programs left: {left}, right: {right}")
    if right is None:
        #print(f"reducer_for_safe_container_all_programs right is None")
        return left 
    
    if isinstance(right,ThreadSafePrograms):
        return right
    
    elif isinstance(right,list):
        merge = left.copy()
        for item in right:
            if len(item) == 2:
                operation = item[0]
                program = item[1]
                if operation == "add":
                    merge.add_program(program.id,program)
                elif operation == "remove":
                    merge.remove_program(program.id)
                elif operation == "update":
                    merge.update_program(program.id,program)
                else:
                    raise ValueError("reducer_for_safe_container_all_programs right must be a tuple of length 2, but you give me a {}".format(len(item)))
        return merge 
    elif isinstance(right,tuple):
        merge = left.copy()
        if len(right) == 2:
            operation = right[0]
            program = right[1]
            if operation == "add":
                merge.add_program(program.id,program)
            elif operation == "remove":
                merge.remove_program(program.id)
            elif operation == "update":
                merge.update_program(program.id,program)
            else:
                raise ValueError("reducer_for_safe_container_all_programs right must be a tuple of length 2, but you give me a {}".format(len(right)))
        
        elif len(right) == 3:
            operation = right[0]
            program_need_replace_id = right[1] #将被替换的程序id
            program_replace_with_program = right[2] #替换的程序对象
            if operation == "replace":
                merge.remove_program(program_need_replace_id)
                merge.add_program(program_replace_with_program.id,program_replace_with_program)
            else:
                raise ValueError("reducer_for_safe_container_all_programs right must be a tuple of length 4, but you give me a {}".format(len(right)))
        else:
            raise ValueError("reducer_for_safe_container_all_programs right must be a tuple of length 2 or 4, but you give me a {}".format(len(right)))
        return merge 

def reducer_IslandState(left:IslandState,right:IslandState)->IslandState:
    '''
    IslandState的更新方式 只需传入IslandState即可
    '''
    return right 