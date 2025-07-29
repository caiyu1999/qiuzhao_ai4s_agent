import os 
import re 
import time 
import asyncio 
import tempfile 
import traceback 
import json 
import base64 
import logging 
from typing import Dict,Any 
from openevolve_graph.program import Program 
from typing import List 
from openevolve_graph.Config.config import Config 
from random import random 
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple, Union
# from openevolve_graph.Graph.Graph_state import BaseModel
from openevolve_graph.Config.config import Config

def pydantic_to_json(model: BaseModel, **kwargs) -> str:
    """
    将Pydantic模型转换为JSON字符串的通用工具函数
    
    Args:
        model: Pydantic模型实例
        **kwargs: 传递给model_dump_json的参数
        
    Returns:
        JSON字符串
        
    Examples:
        # 基本用法
        json_str = pydantic_to_json(my_model)
        
        # 排除None值，美化输出
        json_str = pydantic_to_json(my_model, exclude_none=True, indent=2)
        
        # 排除特定字段
        json_str = pydantic_to_json(my_model, exclude={'password', 'secret'})
    """
    # 过滤掉不适用于model_dump_json的参数
    valid_kwargs = {}
    for key, value in kwargs.items():
        if key in ['exclude_none', 'exclude', 'include', 'by_alias', 'exclude_unset', 'exclude_defaults', 'indent', 'separators', 'default']:
            valid_kwargs[key] = value
    
    default_kwargs = {
        'exclude_none': True,
        'indent': 2
    }
    default_kwargs.update(valid_kwargs)
    return model.model_dump_json(**default_kwargs)

def pydantic_to_dict(model: BaseModel, **kwargs) -> Dict[str, Any]:
    """
    将Pydantic模型转换为字典的通用工具函数
    
    Args:
        model: Pydantic模型实例
        **kwargs: 传递给model_dump的参数
        
    Returns:
        字典表示
        
    Examples:
        # 基本用法
        data_dict = pydantic_to_dict(my_model)
        
        # 排除None值
        data_dict = pydantic_to_dict(my_model, exclude_none=True)
    """
    # 过滤掉不适用于model_dump的参数
    valid_kwargs = {}
    for key, value in kwargs.items():
        if key in ['exclude_none', 'exclude', 'include', 'by_alias', 'exclude_unset', 'exclude_defaults']:
            valid_kwargs[key] = value
    
    default_kwargs = {
        'exclude_none': True
    }
    default_kwargs.update(valid_kwargs)
    return model.model_dump(**default_kwargs)

def save_pydantic_to_file(model: BaseModel, file_path: str, **kwargs) -> None:
    """
    将Pydantic模型保存到JSON文件的通用工具函数
    
    Args:
        model: Pydantic模型实例
        file_path: 文件路径
        **kwargs: 传递给pydantic_to_json的参数
        
    Examples:
        # 基本用法
        save_pydantic_to_file(my_model, "data.json")
        
        # 美化输出
        save_pydantic_to_file(my_model, "data.json", indent=4)
    """
    json_str = pydantic_to_json(model, **kwargs)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json_str)

def format_metrics_safe(metrics: Dict[str, Any]) -> str:
    """
    Safely format metrics dictionary for logging, handling both numeric and string values.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Formatted string representation of metrics
    """
    if not metrics:
        return ""

    formatted_parts = []
    for name, value in metrics.items():
        # Check if value is numeric (int, float)
        if isinstance(value, (int, float)):
            try:
                # Only apply float formatting to numeric values
                formatted_parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):
                # Fallback to string representation if formatting fails
                formatted_parts.append(f"{name}={value}")
        else:
            # For non-numeric values (strings, etc.), just convert to string
            formatted_parts.append(f"{name}={value}")

    return ", ".join(formatted_parts)

logger = logging.getLogger(__name__)


def load_initial_program(path:str)->str:
    # 检查文件是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    with open(path,"r") as f:
        return f.read()
    
    

    
def _feature_coords_to_key(coords: List[int]) -> str:
    """
    将特征坐标转换为字符串键
    
    Args:
        coords: 特征坐标列表
        
    Returns:
        str: 字符串键
    """
    return "-".join(str(c) for c in coords) # e.g [0,1,2,-1] -> "0-1-2--1"


def _calculate_feature_coords(config:Config,state:BaseModel,program: Optional[Program | None | Any]) -> List[int]:
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

    if isinstance(program,Program):
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
                        state.all_programs.values(), min(5, len(state.all_programs))
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
    
    """
Utilities for code parsing, diffing, and manipulation
"""



def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parse evolve blocks from code

    Args:
        code: Source code with evolve blocks

    Returns:
        List of tuples (start_line, end_line, block_content)
    """
    lines = code.split("\n")
    blocks = []

    in_block = False
    start_line = -1
    block_content = []

    for i, line in enumerate(lines):
        if "# EVOLVE-BLOCK-START" in line:
            in_block = True
            start_line = i
            block_content = []
        elif "# EVOLVE-BLOCK-END" in line and in_block:
            in_block = False
            blocks.append((start_line, i, "\n".join(block_content)))
        elif in_block:
            block_content.append(line)

    return blocks


def apply_diff(original_code: str, diff_text: str) -> str:
    """
    Apply a diff to the original code

    Args:
        original_code: Original source code
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        Modified code
    """
    # Split into lines for easier processing
    original_lines = original_code.split("\n")
    result_lines = original_lines.copy()

    # Extract diff blocks
    diff_blocks = extract_diffs(diff_text)

    # Apply each diff block
    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        # Find where the search pattern starts in the original code
        for i in range(len(result_lines) - len(search_lines) + 1):
            if result_lines[i : i + len(search_lines)] == search_lines:
                # Replace the matched section
                result_lines[i : i + len(search_lines)] = replace_lines
                break

    return "\n".join(result_lines)


def extract_diffs(diff_text: str) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text

    Args:
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        List of tuples (search_text, replace_text)
    """
    diff_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
    diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response

    Args:
        llm_response: Response from the LLM
        language: Programming language

    Returns:
        Extracted code or None if not found
    """
    code_block_pattern = r"```" + language + r"\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to any code block
    code_block_pattern = r"```(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to plain text
    return llm_response


def format_diff_summary(diff_blocks: List[Tuple[str, str]]) -> str:
    """
    Create a human-readable summary of the diff

    Args:
        diff_blocks: List of (search_text, replace_text) tuples

    Returns:
        Summary string
    """
    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        # Create a short summary
        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_summary = (
                f"{len(search_lines)} lines" if len(search_lines) > 1 else search_lines[0]
            )
            replace_summary = (
                f"{len(replace_lines)} lines" if len(replace_lines) > 1 else replace_lines[0]
            )
            summary.append(f"Change {i+1}: Replace {search_summary} with {replace_summary}")

    return "\n".join(summary)


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

    

def store_artifacts(
    program_id: str,
    artifacts: Dict[str, Any],
    state: "GraphState",
    config: "Config",
) -> Tuple[str, str]:
    """
    Store artifacts to disk
    Args:
        program_id (str): ID of program
        artifacts (Dict[str, str | bytes]): Artifacts to store
        state (GraphState): Graph state
        config (Config): Config object
    Returns:
        Tuple[str, str]: Path to artifacts JSON and directory
    """
    if not config.enable_artifacts:
        return None, None

    # Create a directory for the program's artifacts
    artifact_dir = os.path.join(config.artifact_dir, program_id)
    os.makedirs(artifact_dir, exist_ok=True)

    # Save artifacts to files and store paths in a JSON file
    artifact_paths = {}
    for key, value in artifacts.items():
        # Handle different artifact types
        path = _artifact_serializer(key, value, artifact_dir)
        if path:
            artifact_paths[key] = path

    # Save the artifact paths to a JSON file
    artifacts_json_path = os.path.join(artifact_dir, "artifacts.json")
    with open(artifacts_json_path, "w") as f:
        json.dump(artifact_paths, f)

    return artifacts_json_path, artifact_dir



def _get_artifact_size(value: Union[str, bytes]) -> int:
    """
    获取工件值的字节大小
    
    Args:
        value: 工件值
        
    Returns:
        int: 字节大小
    """
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    elif isinstance(value, bytes):
        return len(value)
    else:
        return len(str(value).encode("utf-8"))

def _artifact_serializer(key: str, value: Union[str, bytes], artifact_dir: str) -> Optional[str]:
    """
    处理字节的工件JSON序列化器
    
    Args:
        obj: 要序列化的对象
        
    Returns:
        序列化后的对象
    """
    if isinstance(value, bytes):
        safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
        if not safe_key:
            safe_key = "artifact"
        file_path = os.path.join(artifact_dir, safe_key)
        try:
            with open(file_path, "wb") as f:
                f.write(value)
            return file_path
        except Exception as e:
            logger.warning(f"Failed to write artifact {key} to {file_path}: {e}")
            return None
    elif isinstance(value, str):
        safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
        if not safe_key:
            safe_key = "artifact"
        file_path = os.path.join(artifact_dir, safe_key)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(value)
            return file_path
        except Exception as e:
            logger.warning(f"Failed to write artifact {key} to {file_path}: {e}")
            return None
    else:
        # For other types, try to serialize to JSON
        try:
            json_value = json.dumps(value)
            safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
            if not safe_key:
                safe_key = "artifact"
            file_path = os.path.join(artifact_dir, safe_key)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_value)
            return file_path
        except Exception as e:
            logger.warning(f"Failed to serialize artifact {key} to JSON: {e}")
            return None

def _artifact_deserializer(dct):
    """
    处理字节的工件JSON反序列化器
    
    Args:
        dct: 要反序列化的字典
        
    Returns:
        反序列化后的对象
    """
    if "__bytes__" in dct:
        return base64.b64decode(dct["__bytes__"])
    return dct

def _create_artifact_dir(program_id: str,config:Config) -> str:
    """
    为程序创建工件目录
    
    Args:
        program_id: 程序ID
        
    Returns:
        str: 工件目录路径
    """
    base_path = getattr(config, "artifacts_base_path", None)
    if not base_path:
        base_path = (
            os.path.join(config.programs_save_path or ".", "artifacts")
            if config.programs_save_path
            else "./artifacts"
        )

    artifact_dir = os.path.join(base_path, program_id)
    os.makedirs(artifact_dir, exist_ok=True)
    return artifact_dir

def _write_artifact_file(artifact_dir: str, key: str, value: Union[str, bytes]) -> None:
    """
    将工件写入文件
    
    Args:
        artifact_dir: 工件目录
        key: 工件键
        value: 工件值
    """
    # 清理文件名
    safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
    if not safe_key:
        safe_key = "artifact"

    file_path = os.path.join(artifact_dir, safe_key)

    try:
        if isinstance(value, str):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(value)
        elif isinstance(value, bytes):
            with open(file_path, "wb") as f:
                f.write(value)
        else:
            # 转换为字符串并写入
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(value))
    except Exception as e:
        logger.warning(f"Failed to write artifact {key} to {file_path}: {e}")

def _load_artifact_dir(artifact_dir: str) -> Dict[str, Union[str, bytes]]:
    """
    从目录加载工件
    
    Args:
        artifact_dir: 工件目录
        
    Returns:
        Dict[str, Union[str, bytes]]: 工件字典
    """
    artifacts = {}

    try:
        for filename in os.listdir(artifact_dir):
            file_path = os.path.join(artifact_dir, filename)
            if os.path.isfile(file_path):
                try:
                    # 首先尝试作为文本读取
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    artifacts[filename] = content
                except UnicodeDecodeError:
                    # 如果文本失败，作为二进制读取
                    with open(file_path, "rb") as f:
                        content = f.read()
                    artifacts[filename] = content
                except Exception as e:
                    logger.warning(f"Failed to read artifact file {file_path}: {e}")
    except Exception as e:
        logger.warning(f"Failed to list artifact directory {artifact_dir}: {e}")

    return artifacts



def get_top_programs(state:BaseModel, n: int = 10,metric:Optional[str] = None) -> List[Program]:
    """
    获取前N个最优程序 从all_programs中  以metric为指标排序

    Args:
        n: 返回的程序数量
        metric: 用于排序的指标名称（可选，默认使用平均值）

    Returns:
        List[Program]: 前N个最优程序列表
    """
    if not state.all_programs:
        return []

    if metric:
        # 按指定指标排序
        sorted_programs = sorted(
            [p for p in state.all_programs.get_all_programs().values() if metric in p.metrics],
            key=lambda p: p.metrics[metric],
            reverse=True,
        )
    else:
        # 按所有数值指标的平均值排序
        sorted_programs = sorted(
            state.all_programs.get_all_programs().values(),
            key=lambda p: safe_numeric_average(p.metrics),
            reverse=True,
        )

    return sorted_programs[:n]
def _load_artifact_dir(artifact_dir: str) -> Dict[str, Union[str, bytes]]:
        """Load artifacts from a directory"""
        artifacts = {}

        try:
            for filename in os.listdir(artifact_dir):
                file_path = os.path.join(artifact_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        # Try to read as text first
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        artifacts[filename] = content
                    except UnicodeDecodeError:
                        # If text fails, read as binary
                        with open(file_path, "rb") as f:
                            content = f.read()
                        artifacts[filename] = content
                    except Exception as e:
                        logger.warning(f"Failed to read artifact file {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to list artifact directory {artifact_dir}: {e}")

        return artifacts
def get_artifacts(state:BaseModel,program_id: str) -> Dict[str, Union[str, bytes]]:

        program = state.all_programs.get_program(program_id)
        if not program:
            return {}

        artifacts = {}

        # Load small artifacts from JSON
        if program.artifacts_json:
            try:
                small_artifacts = json.loads(program.artifacts_json)
                artifacts.update(small_artifacts)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode artifacts JSON for program {program_id}: {e}")

        # Load large artifacts from disk
        if program.artifact_dir and os.path.exists(program.artifact_dir):
            disk_artifacts = _load_artifact_dir(program.artifact_dir)
            artifacts.update(disk_artifacts)

        return artifacts
    
    
    
    
"""
Safe calculation utilities for metrics containing mixed types
"""

from typing import Any, Dict

def _is_better(program1: Program, program2: Program) -> bool:
        """
        判断program1是否优于program2
        
        比较策略：
        1. 优先使用combined_score
        2. 后备使用所有数值指标的平均值
        3. 如果都没有指标，使用时间戳
        
        Args:
            program1: 第一个程序
            program2: 第二个程序
            
        Returns:
            bool: 如果program1更好则返回True
        """
        # 如果都没有指标，使用最新的
        if not program1.metrics and not program2.metrics:
            return program1.timestamp > program2.timestamp

        # 如果只有一个有指标，它就更好
        if program1.metrics and not program2.metrics:
            return True
        if not program1.metrics and program2.metrics:
            return False

        # 优先检查combined_score（首选指标）
        if "combined_score" in program1.metrics and "combined_score" in program2.metrics:
            return program1.metrics["combined_score"] > program2.metrics["combined_score"]

        # 后备使用所有数值指标的平均值
        avg1 = safe_numeric_average(program1.metrics)
        avg2 = safe_numeric_average(program2.metrics)

        return avg1 > avg2
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


def safe_numeric_sum(metrics: Dict[str, Any]) -> float:
    """
    Calculate the sum of numeric values in a metrics dictionary,
    safely ignoring non-numeric values like strings.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Sum of numeric values, or 0.0 if no numeric values found
    """
    if not metrics:
        return 0.0

    numeric_sum = 0.0
    for value in metrics.values():
        if isinstance(value, (int, float)):
            try:
                # Convert to float and check if it's a valid number
                float_val = float(value)
                if not (float_val != float_val):  # Check for NaN (NaN != NaN is True)
                    numeric_sum += float_val
            except (ValueError, TypeError, OverflowError):
                # Skip invalid numeric values
                continue

    return numeric_sum

if __name__ == "__main__": 
    print(load_initial_program("/Users/caiyu/Desktop/langchain/new_openevolve/examples/circle_packing/initial_program.py"))