# 线程安全容器基类使用指南

## 概述

`ThreadSafeContainer` 是一个通用的线程安全容器基类，提供了完整的并发安全保障。`ThreadSafePrograms` 是基于该基类实现的程序特定容器。

## 架构设计

```
ThreadSafeContainer<K, V>  (抽象基类)
├── ThreadSafeDictContainer<K, V>  (通用字典实现)
└── ThreadSafePrograms  (程序特定实现)
```

## 核心特性

### 🔒 线程安全保障
- 使用 `threading.RLock()` 保护所有操作
- 支持并发读写，避免竞态条件
- 提供原子操作保证

### ⚡ 高性能操作
- 批量操作：`batch_add()`, `batch_remove()`
- 原子更新：`atomic_update()`
- 高效查询：`filter_items()`, `batch_get()`

### 🛠️ 易用接口
- 统一的 API 设计
- 丰富的便利方法
- 类型安全的泛型支持

## 基本使用

### 1. 创建程序容器

```python
from openevolve_graph.utils.thread_safe_programs import create_thread_safe_programs

# 创建程序容器
programs = create_thread_safe_programs()
```

### 2. 添加程序

```python
from new_openevolve.database import Program

# 创建程序
program = Program(
    id="test_prog",
    code="print('Hello, World!')",
    language="python",
    generation=1,
    metrics={"score": 0.8}
)

# 添加程序
programs.add_program("test_prog", program)

# 批量添加
programs.batch_add_programs({
    "prog1": program1,
    "prog2": program2
})
```

### 3. 查询程序

```python
# 获取单个程序
program = programs.get_program("test_prog")

# 获取所有程序
all_programs = programs.get_all_programs()

# 获取程序ID列表
program_ids = programs.get_program_ids()

# 按条件过滤
python_programs = programs.get_programs_by_language("python")
high_score_programs = programs.get_programs_by_metric_range("score", 0.8, 1.0)
```

### 4. 更新和删除

```python
# 删除程序
removed_program = programs.remove_program("test_prog")

# 批量删除
removed_programs = programs.batch_remove_programs(["prog1", "prog2"])

# 原子更新（同时添加和删除）
removed = programs.atomic_update(
    add_programs={"new_prog": new_program},
    remove_program_ids=["old_prog"]
)
```

## 高级功能

### 1. 程序查询

```python
# 按语言查询
python_programs = programs.get_programs_by_language("python")

# 按代数查询
gen1_programs = programs.get_programs_by_generation(1)

# 按父代查询
children = programs.get_programs_with_parent("parent_id")

# 按指标查询
top_programs = programs.get_top_programs_by_metric("score", n=10)
```

### 2. 岛屿管理

```python
# 获取岛屿程序
island_programs = programs.get_programs_by_island(1)

# 迁移程序到岛屿
migrated_count = programs.migrate_programs_to_island(["prog1"], target_island=2)
```

### 3. 统计信息

```python
# 获取程序统计
stats = programs.get_programs_statistics()
print(f"总程序数: {stats['total_programs']}")
print(f"语言分布: {stats['languages']}")
print(f"代数分布: {stats['generations']}")
print(f"指标统计: {stats['metrics']}")
```

### 4. 数据完整性

```python
# 验证程序完整性
integrity = programs.validate_programs_integrity()
if not integrity['is_valid']:
    print("发现问题:")
    for issue in integrity['issues']:
        print(f"  - {issue}")

# 清理过期程序
removed_count = programs.cleanup_stale_programs(
    lambda p: p.generation >= 10  # 保留代数>=10的程序
)
```

## 并发安全使用

### 1. 使用辅助类

```python
from openevolve_graph.utils.thread_safe_programs import ProgramOperations

# 安全操作
ProgramOperations.safe_add_program(programs, "id", program)
ProgramOperations.safe_batch_add_programs(programs, program_dict)
ProgramOperations.safe_atomic_update(programs, add_programs, remove_ids)
```

### 2. 在 LangGraph 中使用

```python
from openevolve_graph.Graph.Graph_state import GraphState

def parallel_node_1(state: GraphState) -> GraphState:
    """并行节点1"""
    # 安全地添加程序
    ProgramOperations.safe_add_program(
        state.all_programs, 
        "new_prog", 
        new_program
    )
    return state

def parallel_node_2(state: GraphState) -> GraphState:
    """并行节点2"""
    # 安全地获取程序
    program = state.all_programs.get_program("some_id")
    return state
```

## 自定义容器

### 1. 继承基类

```python
from openevolve_graph.utils.thread_safe_container import ThreadSafeContainer

@dataclass
class CustomContainer(ThreadSafeContainer[str, CustomType]):
    """自定义线程安全容器"""
    
    _data: Dict[str, CustomType] = field(default_factory=dict, init=False)
    
    def _internal_add(self, key: str, value: CustomType) -> None:
        self._data[key] = value
    
    def _internal_remove(self, key: str) -> Optional[CustomType]:
        return self._data.pop(key, None)
    
    # ... 实现其他抽象方法
```

### 2. 添加特定方法

```python
class CustomContainer(ThreadSafeContainer[str, CustomType]):
    # ... 基本实现
    
    def custom_query(self, condition: str) -> List[CustomType]:
        """自定义查询方法"""
        return self.filter_items(lambda item: item.matches(condition))
    
    def custom_operation(self, operation: str) -> int:
        """自定义操作方法"""
        with self._lock:  # 使用基类的锁
            # 自定义逻辑
            return result
```

## 性能优化

### 1. 批量操作

```python
# 好的做法：批量操作
programs.batch_add_programs({
    "prog1": program1,
    "prog2": program2,
    "prog3": program3
})

# 避免：多次单独操作
# programs.add_program("prog1", program1)
# programs.add_program("prog2", program2)
# programs.add_program("prog3", program3)
```

### 2. 原子更新

```python
# 好的做法：原子更新
programs.atomic_update(
    add_programs=new_programs,
    remove_program_ids=old_program_ids
)

# 避免：分步操作
# programs.batch_remove_programs(old_program_ids)
# programs.batch_add_programs(new_programs)
```

### 3. 合理使用锁

```python
# 使用上下文管理器进行复杂操作
with programs._atomic_operation():
    # 复杂的多步操作
    program = programs._internal_get("some_id")
    if program:
        program.metrics["score"] += 0.1
        programs._internal_add("some_id", program)
```

## 错误处理

### 1. 异常处理

```python
try:
    program = programs.get_program("nonexistent_id")
except ValueError as e:
    print(f"程序不存在: {e}")

# 或者使用安全方法
program = programs.get_program("id")
if program is None:
    print("程序不存在")
```

### 2. 数据验证

```python
# 在添加前验证
if program.id and program.code:
    programs.add_program(program.id, program)
else:
    print("程序数据不完整")
```

## 最佳实践

1. **优先使用批量操作**：提高性能，减少锁竞争
2. **使用原子更新**：确保数据一致性
3. **合理使用辅助类**：简化代码，提高可读性
4. **定期验证数据完整性**：及时发现和修复问题
5. **清理过期数据**：保持良好的内存使用

## 测试

运行测试验证功能：

```bash
python openevolve_graph/test_base_class.py
```

这将验证：
- 基类功能正确性
- 继承实现正确性
- 线程安全性
- 程序特定功能
- 数据完整性

## 总结

通过使用 `ThreadSafeContainer` 基类设计，您可以：

1. **安全地进行并发操作**：无需担心竞态条件
2. **提高性能**：通过批量和原子操作
3. **简化代码**：统一的API和丰富的便利方法
4. **确保数据完整性**：内置的验证和清理机制
5. **易于扩展**：基于抽象基类的设计模式

这个设计完美解决了多个并行子图同时修改 `all_programs` 的并发安全问题，同时提供了丰富的功能和良好的性能。 