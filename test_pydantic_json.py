#!/usr/bin/env python3
"""
测试Pydantic模型的JSON序列化方法
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import json
from enum import Enum

# 示例枚举
class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

# 示例Pydantic模型
class User(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None
    status: Status = Status.ACTIVE
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Config(BaseModel):
    debug: bool = False
    timeout: int = 30
    users: List[User] = Field(default_factory=list)

def test_pydantic_json_methods():
    """测试不同的Pydantic JSON序列化方法"""
    
    # 创建示例数据
    user1 = User(id=1, name="张三", email="zhangsan@example.com", age=25)
    user2 = User(id=2, name="李四", email="lisi@example.com", status=Status.INACTIVE)
    
    config = Config(
        debug=True,
        timeout=60,
        users=[user1, user2]
    )
    
    print("=== Pydantic模型JSON序列化方法演示 ===\n")
    
    # 方法1: 使用 model_dump_json() (推荐)
    print("1. 使用 model_dump_json() (推荐方法):")
    json_str1 = config.model_dump_json(exclude_none=True, indent=2)
    print(json_str1)
    print()
    
    # 方法2: 使用 model_dump() 然后 json.dumps()
    print("2. 使用 model_dump() + json.dumps():")
    data_dict = config.model_dump(exclude_none=True, mode='json')  # 使用json模式
    json_str2 = json.dumps(data_dict, indent=2, ensure_ascii=False)
    print(json_str2)
    print()
    
    # 方法3: 自定义to_json方法
    print("3. 自定义to_json方法:")
    json_str3 = config.model_dump_json(
        exclude_none=True, 
        indent=2,
        exclude={'debug'}  # 排除debug字段
    )
    print(json_str3)
    print()
    
    # 方法4: 保存到文件
    print("4. 保存到文件:")
    with open("test_config.json", "w", encoding="utf-8") as f:
        f.write(config.model_dump_json(exclude_none=True, indent=2))
    print("已保存到 test_config.json")
    print()
    
    # 方法5: 从JSON反序列化
    print("5. 从JSON反序列化:")
    loaded_config = Config.model_validate_json(json_str1)
    print(f"反序列化成功: {loaded_config.debug}, {len(loaded_config.users)} 个用户")
    print()

def test_enum_serialization():
    """测试枚举序列化"""
    print("=== 枚举序列化测试 ===")
    
    user = User(id=1, name="测试用户", email="test@example.com", status=Status.INACTIVE)
    
    # Pydantic会自动将枚举转换为字符串
    json_str = user.model_dump_json(indent=2)
    print("枚举自动序列化为字符串:")
    print(json_str)
    print()

def test_complex_objects():
    """测试复杂对象序列化"""
    print("=== 复杂对象序列化测试 ===")
    
    # 包含复杂嵌套结构的模型
    class ComplexModel(BaseModel):
        name: str
        data: Dict[str, Any]
        nested_list: List[Dict[str, Any]]
        
    complex_obj = ComplexModel(
        name="复杂对象",
        data={
            "number": 42,
            "string": "hello",
            "boolean": True,
            "null_value": None
        },
        nested_list=[
            {"id": 1, "value": "first"},
            {"id": 2, "value": "second"}
        ]
    )
    
    json_str = complex_obj.model_dump_json(indent=2, exclude_none=True)
    print("复杂对象序列化:")
    print(json_str)
    print()

if __name__ == "__main__":
    test_pydantic_json_methods()
    test_enum_serialization()
    test_complex_objects()
    
    print("=== 总结 ===")
    print("Pydantic v2 推荐的JSON序列化方法:")
    print("1. model_dump_json() - 直接转换为JSON字符串")
    print("2. model_dump() - 转换为字典，然后使用json.dumps()")
    print("3. 自动处理枚举、嵌套对象、None值等")
    print("4. 支持exclude、include、exclude_none等参数") 