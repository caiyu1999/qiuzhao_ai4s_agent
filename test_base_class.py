"""
测试线程安全容器基类设计

验证 ThreadSafeContainer 基类和 ThreadSafePrograms 子类的正确性
"""

import threading
import time
import random
from typing import Dict, List
from openevolve_graph.utils.thread_safe_container import ThreadSafeContainer, ThreadSafeDictContainer, ContainerValidator
from openevolve_graph.utils.thread_safe_container import ThreadSafePrograms, ProgramOperations, create_thread_safe_programs
from new_openevolve.database import Program


def test_base_class_functionality():
    """测试基类功能"""
    print("=== 基类功能测试 ===")
    
    # 测试通用字典容器
    container = ThreadSafeDictContainer[str, str]()
    
    # 基本操作测试
    container.add("key1", "value1")
    container.add("key2", "value2")
    
    assert container.get("key1") == "value1"
    assert len(container) == 2
    assert "key1" in container
    assert "key3" not in container
    
    # 批量操作测试
    batch_data = {"key3": "value3", "key4": "value4"}
    container.batch_add(batch_data)
    assert len(container) == 4
    
    # 原子更新测试
    removed = container.atomic_update(
        add_items={"key5": "value5"},
        remove_keys=["key1"]
    )
    assert "key1" in removed
    assert len(container) == 4
    
    # 过滤测试
    filtered = container.filter_items(lambda v: v.endswith("3"))
    assert len(filtered) == 1
    assert "key3" in filtered
    
    print("基类功能测试通过 ✓")


def test_program_container_functionality():
    """测试程序容器功能"""
    print("\n=== 程序容器功能测试 ===")
    
    programs = create_thread_safe_programs()
    
    # 创建测试程序
    program1 = Program(
        id="prog1",
        code="print('hello')",
        language="python",
        generation=1,
        metrics={"score": 0.8, "complexity": 10}
    )
    
    program2 = Program(
        id="prog2",
        code="print('world')",
        language="python",
        generation=2,
        metrics={"score": 0.6, "complexity": 5},
        parent_id="prog1"
    )
    
    program3 = Program(
        id="prog3",
        code="console.log('js')",
        language="javascript",
        generation=1,
        metrics={"score": 0.9, "complexity": 8}
    )
    
    # 添加程序
    programs.add_program("prog1", program1)
    programs.add_program("prog2", program2)
    programs.add_program("prog3", program3)
    
    assert len(programs) == 3
    
    # 测试程序特定方法
    python_programs = programs.get_programs_by_language("python")
    assert len(python_programs) == 2
    
    gen1_programs = programs.get_programs_by_generation(1)
    assert len(gen1_programs) == 2
    
    children = programs.get_programs_with_parent("prog1")
    assert len(children) == 1
    assert "prog2" in children
    
    # 测试指标范围查询
    high_score_programs = programs.get_programs_by_metric_range("score", 0.7, 1.0)
    assert len(high_score_programs) == 2
    
    # 测试顶级程序
    top_programs = programs.get_top_programs_by_metric("score", n=2)
    assert len(top_programs) == 2
    assert top_programs[0].id == "prog3"  # 最高分
    
    # 测试统计信息
    stats = programs.get_programs_statistics()
    assert stats['total_programs'] == 3
    assert stats['languages']['python'] == 2
    assert stats['languages']['javascript'] == 1
    assert stats['generations'][1] == 2
    assert stats['generations'][2] == 1
    
    print("程序容器功能测试通过 ✓")


def test_thread_safety_with_inheritance():
    """测试继承后的线程安全性"""
    print("\n=== 继承线程安全性测试 ===")
    
    programs = create_thread_safe_programs()
    results = {"add_count": 0, "remove_count": 0, "errors": []}
    results_lock = threading.Lock()
    
    def worker_add(worker_id: int, num_programs: int):
        """添加程序的工作线程"""
        try:
            for i in range(num_programs):
                program = Program(
                    id=f"worker_{worker_id}_prog_{i}",
                    code=f"print('Worker {worker_id} Program {i}')",
                    language="python",
                    generation=i,
                    metrics={"score": random.random()}
                )
                programs.add_program(program.id, program)
                time.sleep(0.001)
            
            with results_lock:
                results["add_count"] += num_programs
        except Exception as e:
            with results_lock:
                results["errors"].append(f"Worker {worker_id} add error: {e}")
    
    def worker_read(worker_id: int, num_reads: int):
        """读取程序的工作线程"""
        try:
            for i in range(num_reads):
                # 测试不同的读取操作
                programs.get_all_programs()
                programs.get_program_ids()
                programs.get_programs_by_language("python")
                programs.get_programs_statistics()
                time.sleep(0.001)
        except Exception as e:
            with results_lock:
                results["errors"].append(f"Worker {worker_id} read error: {e}")
    
    def worker_remove(worker_id: int):
        """移除程序的工作线程"""
        try:
            time.sleep(0.5)  # 等待一些程序被添加
            
            all_ids = programs.get_program_ids()
            if all_ids:
                # 移除一些程序
                remove_ids = all_ids[:min(5, len(all_ids))]
                removed = programs.batch_remove_programs(remove_ids)
                
                with results_lock:
                    results["remove_count"] += len(removed)
        except Exception as e:
            with results_lock:
                results["errors"].append(f"Worker {worker_id} remove error: {e}")
    
    # 启动多个线程
    threads = []
    
    # 添加程序的线程
    for i in range(3):
        thread = threading.Thread(target=worker_add, args=(i, 10))
        threads.append(thread)
        thread.start()
    
    # 读取程序的线程
    for i in range(3):
        thread = threading.Thread(target=worker_read, args=(i, 20))
        threads.append(thread)
        thread.start()
    
    # 移除程序的线程
    for i in range(2):
        thread = threading.Thread(target=worker_remove, args=(i,))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 验证结果
    print(f"添加程序数量: {results['add_count']}")
    print(f"移除程序数量: {results['remove_count']}")
    print(f"最终程序数量: {len(programs)}")
    
    if results["errors"]:
        print(f"错误数量: {len(results['errors'])}")
        for error in results["errors"]:
            print(f"  - {error}")
        return False
    else:
        print("继承线程安全性测试通过 ✓")
        return True


def test_program_operations_helper():
    """测试程序操作辅助类"""
    print("\n=== 程序操作辅助类测试 ===")
    
    programs = create_thread_safe_programs()
    
    # 测试辅助方法
    program1 = Program(
        id="test1",
        code="print('test1')",
        language="python",
        metrics={"combined_score": 0.8}
    )
    
    program2 = Program(
        id="test2",
        code="print('test2')",
        language="python",
        metrics={"combined_score": 0.9}
    )
    
    # 使用辅助方法添加程序
    ProgramOperations.safe_add_program(programs, "test1", program1)
    ProgramOperations.safe_batch_add_programs(programs, {"test2": program2})
    
    assert len(programs) == 2
    
    # 测试获取最佳程序
    best_programs = ProgramOperations.get_best_programs(programs, "combined_score", 2)
    assert len(best_programs) == 2
    assert best_programs[0].id == "test2"  # 分数更高
    
    # 测试原子更新
    new_program = Program(
        id="test3",
        code="print('test3')",
        language="python",
        metrics={"combined_score": 0.7}
    )
    
    removed = ProgramOperations.safe_atomic_update(
        programs,
        add_programs={"test3": new_program},
        remove_program_ids=["test1"]
    )
    
    assert "test1" in removed
    assert len(programs) == 2
    assert programs.get_program("test3") is not None
    
    print("程序操作辅助类测试通过 ✓")


def test_island_functionality():
    """测试岛屿相关功能"""
    print("\n=== 岛屿功能测试 ===")
    
    programs = create_thread_safe_programs()
    
    # 创建带岛屿信息的程序
    program1 = Program(
        id="island1_prog1",
        code="print('island1')",
        language="python",
        metadata={"island": 1}
    )
    
    program2 = Program(
        id="island2_prog1",
        code="print('island2')",
        language="python",
        metadata={"island": 2}
    )
    
    program3 = Program(
        id="island1_prog2",
        code="print('island1 again')",
        language="python",
        metadata={"island": 1}
    )
    
    programs.add_program("island1_prog1", program1)
    programs.add_program("island2_prog1", program2)
    programs.add_program("island1_prog2", program3)
    
    # 测试岛屿程序获取
    island1_programs = programs.get_programs_by_island(1)
    assert len(island1_programs) == 2
    
    island2_programs = programs.get_programs_by_island(2)
    assert len(island2_programs) == 1
    
    # 测试程序迁移
    migrated = programs.migrate_programs_to_island(["island1_prog1"], 3)
    assert migrated == 1
    
    # 验证迁移结果
    migrated_program = programs.get_program("island1_prog1")
    assert migrated_program is not None
    assert migrated_program.metadata["island"] == 3
    
    print("岛屿功能测试通过 ✓")


def test_data_integrity():
    """测试数据完整性"""
    print("\n=== 数据完整性测试 ===")
    
    programs = create_thread_safe_programs()
    
    # 添加正常程序
    good_program = Program(
        id="good_prog",
        code="print('good')",
        language="python",
        parent_id=None
    )
    
    # 添加有问题的程序
    bad_program1 = Program(
        id="bad_prog1",
        code="",  # 空代码
        language="python"
    )
    
    bad_program2 = Program(
        id="bad_prog2",
        code="print('bad')",
        language="python",
        parent_id="nonexistent_parent"  # 不存在的父代
    )
    
    programs.add_program("good_prog", good_program)
    programs.add_program("bad_prog1", bad_program1)
    programs.add_program("bad_prog2", bad_program2)
    
    # 验证完整性
    integrity_result = programs.validate_programs_integrity()
    
    assert not integrity_result["is_valid"]
    assert len(integrity_result["issues"]) == 2
    assert integrity_result["total_programs"] == 3
    
    print("数据完整性测试通过 ✓")


def run_all_tests():
    """运行所有测试"""
    print("开始测试线程安全容器基类设计...\n")
    
    test_results = []
    
    try:
        test_base_class_functionality()
        test_results.append(True)
    except Exception as e:
        print(f"基类功能测试失败: {e}")
        test_results.append(False)
    
    try:
        test_program_container_functionality()
        test_results.append(True)
    except Exception as e:
        print(f"程序容器功能测试失败: {e}")
        test_results.append(False)
    
    try:
        result = test_thread_safety_with_inheritance()
        test_results.append(result)
    except Exception as e:
        print(f"继承线程安全性测试失败: {e}")
        test_results.append(False)
    
    try:
        test_program_operations_helper()
        test_results.append(True)
    except Exception as e:
        print(f"程序操作辅助类测试失败: {e}")
        test_results.append(False)
    
    try:
        test_island_functionality()
        test_results.append(True)
    except Exception as e:
        print(f"岛屿功能测试失败: {e}")
        test_results.append(False)
    
    try:
        test_data_integrity()
        test_results.append(True)
    except Exception as e:
        print(f"数据完整性测试失败: {e}")
        test_results.append(False)
    
    # 总结结果
    print(f"\n=== 测试总结 ===")
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("所有测试通过！基类设计正确 ✓")
    else:
        print("存在测试失败，请检查代码 ✗")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 