import os 
import re 
from typing import Dict,Any 
from openevolve_graph.program import Program 
from typing import List 
from openevolve_graph.Config.config import Config 
from random import random 
from pydantic import BaseModel

def load_initial_program(path:str)->str:
    # 检查文件是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    with open(path,"r") as f:
        return f.read()
    
    

def extract_code_language(code: str) -> str:
    """
    Try to determine the language of a code snippet

    Args:
        code: Code snippet

    Returns:
        Detected language or "unknown"
    """
    # Look for common language signatures
    if re.search(r"^(import|from|def|class)\s", code, re.MULTILINE):
        return "python"
    elif re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        return "java"
    elif re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        return "sql"

    return "unknown"
    
def _feature_coords_to_key(coords: List[int]) -> str:
    """
    将特征坐标转换为字符串键
    
    Args:
        coords: 特征坐标列表
        
    Returns:
        str: 字符串键
    """
    return "-".join(str(c) for c in coords)


def _calculate_feature_coords(config:Config,state:BaseModel,program: Program) -> List[int]:
    """
    计算程序在MAP-Elites特征网格中的坐标
    
    支持多种特征维度：
    - complexity: 基于代码长度的复杂度
    - diversity: 基于与其他程序的编辑距离
    - score: 基于数值指标的平均值
    - 特定指标: 直接使用程序的指标值
    
    Args:
        program: 要计算特征的程序
        
    Returns:
        List[int]: 特征坐标列表
    """
    coords = []

    for dim in config.island.feature_dimensions:
        if dim == "complexity":
            # 使用代码长度作为复杂度度量
            complexity = len(program.code)
            bin_idx = min(int(complexity / 1000 * config.island.feature_bins), config.island.feature_bins - 1)
            coords.append(bin_idx)
        elif dim == "diversity":
            # 使用与其他程序的平均编辑距离
            if len(state.all_programs) < 5:
                bin_idx = 0
            else:
                sample_programs = random.sample(
                    list(state.all_programs.values()), min(5, len(state.all_programs))
                )
                avg_distance = sum(
                    calculate_edit_distance(program.code, other.code)
                    for other in sample_programs
                ) / len(sample_programs)
                bin_idx = min(
                    int(avg_distance / 1000 * config.island.feature_bins), config.island.feature_bins - 1
                )
            coords.append(bin_idx)
        elif dim == "score":
            # 使用数值指标的平均值
            if not program.metrics:
                bin_idx = 0
            else:
                avg_score = safe_numeric_average(program.metrics)
                bin_idx = min(int(avg_score * config.island.feature_bins), config.island.feature_bins - 1)
            coords.append(bin_idx)
        elif dim in program.metrics:
            # 使用特定指标
            score = program.metrics[dim]
            bin_idx = min(int(score * config.island.feature_bins), config.island.feature_bins - 1)
            coords.append(bin_idx)
        else:
            # 如果未找到特征，默认使用中间分箱
            coords.append(config.island.feature_bins // 2)

    return coords


def calculate_edit_distance(code1: str, code2: str) -> int:
    """
    Calculate the Levenshtein edit distance between two code snippets

    Args:
        code1: First code snippet
        code2: Second code snippet

    Returns:
        Edit distance (number of operations needed to transform code1 into code2)
    """
    if code1 == code2:
        return 0

    # Simple implementation of Levenshtein distance
    m, n = len(code1), len(code2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if code1[i - 1] == code2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[m][n]



def safe_numeric_average(metrics: Dict[str, Any]) -> float:
    """
    Calculate the average of numeric values in a metrics dictionary,
    safely ignoring non-numeric values like strings.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Average of numeric values, or 0.0 if no numeric values found
    """
    if not metrics:
        return 0.0

    numeric_values = []
    for value in metrics.values():
        if isinstance(value, (int, float)):
            try:
                # Convert to float and check if it's a valid number
                float_val = float(value)
                if not (float_val != float_val):  # Check for NaN (NaN != NaN is True)
                    numeric_values.append(float_val)
            except (ValueError, TypeError, OverflowError):
                # Skip invalid numeric values
                continue

    if not numeric_values:
        return 0.0

    return sum(numeric_values) / len(numeric_values)
    
    
    
    
if __name__ == "__main__": 
    print(load_initial_program("/Users/caiyu/Desktop/langchain/new_openevolve/examples/circle_packing/initial_program.py"))