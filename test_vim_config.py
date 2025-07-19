#!/usr/bin/env python3
"""
测试vim Python配置的示例文件
"""

import os
import sys
from typing import List, Dict, Optional


class VimConfigTest:
    """测试vim配置的示例类"""
    
    def __init__(self, name: str):
        self.name = name
        self.data: List[int] = []
    
    def add_data(self, value: int) -> None:
        """添加数据到列表"""
        self.data.append(value)
    
    def get_sum(self) -> int:
        """计算数据总和"""
        return sum(self.data)
    
    def process_data(self, multiplier: float = 1.0) -> List[float]:
        """处理数据"""
        return [x * multiplier for x in self.data]


def main():
    """主函数"""
    test = VimConfigTest("测试")
    
    # 添加一些测试数据
    for i in range(1, 6):
        test.add_data(i)
    
    print(f"数据总和: {test.get_sum()}")
    print(f"处理后的数据: {test.process_data(2.0)}")


if __name__ == "__main__":
    main() 