"""
并发安全性测试
测试 ThreadSafePrograms 类在多线程环境下的安全性
"""

import threading
import time
import random
from typing import List, Dict
from openevolve_graph.Graph.Graph_state import ThreadSafePrograms, Program

def test_concurrent_operations():
    """测试并发操作的安全性"""
    
    # 创建线程安全的程序容器
    safe_programs = ThreadSafePrograms()
    
    # 测试结果统计
    results = {
        'add_success': 0,
        'remove_success': 0,
        'read_success': 0,
        'errors': []
    }
    results_lock = threading.Lock()
    
    def worker_add_programs(worker_id: int, num_programs: int):
        """工作线程：添加程序"""
        try:
            for i in range(num_programs):
                program_id = f"worker_{worker_id}_program_{i}"
                program = Program(
                    id=program_id,
                    code=f"print('Worker {worker_id} Program {i}')",
                    language="python"
                )
                safe_programs.add_program(program_id, program)
                time.sleep(0.001)  # 模拟一些处理时间
            
            with results_lock:
                results['add_success'] += num_programs
        except Exception as e:
            with results_lock:
                results['errors'].append(f"Worker {worker_id} add error: {e}")
    
    def worker_remove_programs(worker_id: int, program_ids: List[str]):
        """工作线程：移除程序"""
        try:
            removed_count = 0
            for program_id in program_ids:
                if safe_programs.remove_program(program_id):
                    removed_count += 1
                time.sleep(0.001)
            
            with results_lock:
                results['remove_success'] += removed_count
        except Exception as e:
            with results_lock:
                results['errors'].append(f"Worker {worker_id} remove error: {e}")
    
    def worker_read_programs(worker_id: int, iterations: int):
        """工作线程：读取程序"""
        try:
            read_count = 0
            for _ in range(iterations):
                # 随机选择读取操作
                operation = random.choice(['get_all', 'get_ids', 'len', 'contains'])
                
                if operation == 'get_all':
                    programs = safe_programs.get_all_programs()
                    read_count += len(programs)
                elif operation == 'get_ids':
                    ids = safe_programs.get_program_ids()
                    read_count += len(ids)
                elif operation == 'len':
                    length = len(safe_programs)
                    read_count += length
                elif operation == 'contains':
                    # 随机检查一个程序是否存在
                    test_id = f"worker_0_program_0"
                    if test_id in safe_programs:
                        read_count += 1
                
                time.sleep(0.001)
            
            with results_lock:
                results['read_success'] += read_count
        except Exception as e:
            with results_lock:
                results['errors'].append(f"Worker {worker_id} read error: {e}")
    
    def worker_batch_operations(worker_id: int):
        """工作线程：批量操作"""
        try:
            # 批量添加
            batch_programs = {}
            for i in range(5):
                program_id = f"batch_worker_{worker_id}_program_{i}"
                batch_programs[program_id] = Program(
                    id=program_id,
                    code=f"print('Batch Worker {worker_id} Program {i}')",
                    language="python"
                )
            
            safe_programs.batch_add_programs(batch_programs)
            
            # 批量移除一些程序
            remove_ids = [f"batch_worker_{worker_id}_program_{i}" for i in range(2)]
            removed = safe_programs.batch_remove_programs(remove_ids)
            
            with results_lock:
                results['add_success'] += len(batch_programs)
                results['remove_success'] += len(removed)
        except Exception as e:
            with results_lock:
                results['errors'].append(f"Batch worker {worker_id} error: {e}")
    
    # 创建多个线程进行并发测试
    threads = []
    
    # 添加程序的线程
    for i in range(3):
        thread = threading.Thread(target=worker_add_programs, args=(i, 10))
        threads.append(thread)
    
    # 读取程序的线程
    for i in range(3):
        thread = threading.Thread(target=worker_read_programs, args=(i, 20))
        threads.append(thread)
    
    # 批量操作的线程
    for i in range(2):
        thread = threading.Thread(target=worker_batch_operations, args=(i,))
        threads.append(thread)
    
    # 启动所有线程
    print("Starting concurrent operations test...")
    start_time = time.time()
    
    for thread in threads:
        thread.start()
    
    # 等待一段时间后启动移除线程
    time.sleep(0.5)
    
    # 获取当前程序ID列表，然后启动移除线程
    current_ids = safe_programs.get_program_ids()
    remove_ids = current_ids[:min(10, len(current_ids))]
    
    for i in range(2):
        thread = threading.Thread(target=worker_remove_programs, args=(i, remove_ids))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    
    # 输出测试结果
    print(f"\n=== 并发测试结果 ===")
    print(f"测试时间: {end_time - start_time:.2f}秒")
    print(f"添加成功: {results['add_success']}")
    print(f"移除成功: {results['remove_success']}")
    print(f"读取成功: {results['read_success']}")
    print(f"最终程序数量: {len(safe_programs)}")
    
    if results['errors']:
        print(f"错误数量: {len(results['errors'])}")
        for error in results['errors'][:5]:  # 只显示前5个错误
            print(f"  - {error}")
    else:
        print("没有错误发生 ✓")
    
    # 验证数据一致性
    final_programs = safe_programs.get_all_programs()
    final_ids = safe_programs.get_program_ids()
    
    print(f"\n=== 数据一致性检查 ===")
    print(f"程序字典长度: {len(final_programs)}")
    print(f"程序ID列表长度: {len(final_ids)}")
    print(f"len()方法结果: {len(safe_programs)}")
    
    # 检查是否一致
    if len(final_programs) == len(final_ids) == len(safe_programs):
        print("数据一致性检查通过 ✓")
    else:
        print("数据一致性检查失败 ✗")
    
    return len(results['errors']) == 0

def test_atomic_operations():
    """测试原子操作"""
    print("\n=== 原子操作测试 ===")
    
    safe_programs = ThreadSafePrograms()
    
    # 先添加一些程序
    initial_programs = {}
    for i in range(10):
        program_id = f"initial_program_{i}"
        initial_programs[program_id] = Program(
            id=program_id,
            code=f"print('Initial Program {i}')",
            language="python"
        )
    
    safe_programs.batch_add_programs(initial_programs)
    print(f"初始程序数量: {len(safe_programs)}")
    
    # 测试原子更新
    add_programs = {}
    for i in range(5):
        program_id = f"new_program_{i}"
        add_programs[program_id] = Program(
            id=program_id,
            code=f"print('New Program {i}')",
            language="python"
        )
    
    remove_ids = [f"initial_program_{i}" for i in range(3)]
    
    # 执行原子更新
    removed_programs = safe_programs.atomic_update(add_programs, remove_ids)
    
    print(f"添加了 {len(add_programs)} 个程序")
    print(f"移除了 {len(removed_programs)} 个程序")
    print(f"最终程序数量: {len(safe_programs)}")
    
    # 验证操作结果
    expected_count = 10 + 5 - 3  # 初始10个 + 添加5个 - 移除3个
    actual_count = len(safe_programs)
    
    if actual_count == expected_count:
        print("原子操作测试通过 ✓")
        return True
    else:
        print(f"原子操作测试失败 ✗ (期望: {expected_count}, 实际: {actual_count})")
        return False

if __name__ == "__main__":
    print("开始线程安全性测试...")
    
    # 运行并发测试
    concurrent_test_passed = test_concurrent_operations()
    
    # 运行原子操作测试
    atomic_test_passed = test_atomic_operations()
    
    print(f"\n=== 总体测试结果 ===")
    if concurrent_test_passed and atomic_test_passed:
        print("所有测试通过 ✓")
    else:
        print("存在测试失败 ✗")
        if not concurrent_test_passed:
            print("  - 并发测试失败")
        if not atomic_test_passed:
            print("  - 原子操作测试失败") 