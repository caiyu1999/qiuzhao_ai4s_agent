import json
from enum import Enum
from openevolve_graph.visualization.socket_sc import ProgramJSONEncoder, safe_serialize

class TestStatus(Enum):
    SAMPLE = "sample"
    EVALUATE_CHILD = "evaluate_child"
    BUILD_PROMPT = "build_prompt"

# 测试枚举序列化
test_data = {
    "status": TestStatus.SAMPLE,
    "message": "test"
}

print("原始数据:", test_data)
print("序列化结果:", safe_serialize(test_data))

# 测试反序列化
serialized = safe_serialize(test_data)
deserialized = json.loads(serialized)
print("反序列化结果:", deserialized) 