

from openevolve_graph.program import Program
from openevolve_graph.Config import Config
from typing import Dict,Any,Callable
import os 
import sys 
import importlib.util
import logging
import time 
#logger = logging.get#logger(__name__)
import tempfile
import asyncio
import traceback 
from openevolve_graph.Graph.Graph_state import GraphState
from typing import Union 


"""
Evaluation result structures for OpenEvolve
"""

import json
from dataclasses import dataclass, field,Field
from typing import Dict, Union


@dataclass
class EvaluationResult:
    """
    Result of program evaluation containing both metrics and optional artifacts

    This maintains backward compatibility with the existing dict[str, float] contract
    while adding a side-channel for arbitrary artifacts (text or binary data).
    """

    metrics: Dict[str, float] = field(default_factory=dict) # mandatory - existing contract
    artifacts: Dict[str, Union[str, bytes | bool | Any]] = field(default_factory=dict)  # optional side-channel

    @classmethod
    def from_dict(cls, metrics: Dict[str, float]) -> "EvaluationResult":
        """Auto-wrap dict returns for backward compatibility"""
        if isinstance(metrics, dict):
            # 确保所有metrics值都是float类型
            cleaned_metrics = {}
            for key, value in metrics.items():
                try:
                    if isinstance(value, (int, float)):
                        cleaned_metrics[key] = float(value)
                    elif isinstance(value, str):
                        # 尝试将字符串转换为float，如果失败则设为0.0
                        try:
                            cleaned_metrics[key] = float(value)
                        except (ValueError, TypeError):
                            cleaned_metrics[key] = 0.0
                    else:
                        cleaned_metrics[key] = 0.0
                except Exception:
                    cleaned_metrics[key] = 0.0
            return cls(metrics=cleaned_metrics)
       

    def to_dict(self) -> Dict[str, float]:
        """Backward compatibility - return just metrics"""
        return self.metrics

    def has_artifacts(self) -> bool:
        """Check if this result contains any artifacts"""
        return bool(self.artifacts)

    def get_artifact_keys(self) -> list:
        """Get list of artifact keys"""
        return list(self.artifacts.keys())

    def get_artifact_size(self, key: str) -> int:
        """Get size of a specific artifact in bytes"""
        if key not in self.artifacts:
            return 0

        value = self.artifacts[key]
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        else:
            return len(str(value).encode("utf-8"))

    def get_total_artifact_size(self) -> int:
        """Get total size of all artifacts in bytes"""
        return sum(self.get_artifact_size(key) for key in self.artifacts.keys())

def _load_evaluation_function(evaluation_file:str) -> Callable:
        """Load the evaluation function from the evaluation file"""
        if not os.path.exists(evaluation_file):
            raise ValueError(f"Evaluation file {evaluation_file} not found")

        try:
            # Add the evaluation file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                #logger.debug(f"Added {eval_dir} to Python path for local imports")

            spec = importlib.util.spec_from_file_location("evaluation_module", evaluation_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {evaluation_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["evaluation_module"] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "evaluate"):
                raise AttributeError(
                    f"Evaluation file {evaluation_file} does not contain an 'evaluate' function"
                )

            evaluate_function = module.evaluate
            #logger.info(f"Successfully loaded evaluation function from {evaluation_file}")
            return evaluate_function
        except Exception as e:
            #logger.error(f"Error loading evaluation function: {str(e)}")
            raise

def passes_threshold(metrics: Dict[str, float], threshold: float) -> bool:
    """
    Check if metrics pass a threshold

    Args:
        metrics: Dictionary of metric name to score
        threshold: Threshold to pass

    Returns:
        True if metrics pass threshold
    """
    if not metrics:
        return False

    # Calculate average score, skipping non-numeric values and 'error' key
    valid_metrics = []
    for name, value in metrics.items():
        # Skip 'error' keys and ensure values are numeric
        if name != "error" and isinstance(value, (int, float)):
            try:
                valid_metrics.append(float(value))
            except (TypeError, ValueError):
                #logger.warning(f"Skipping non-numeric metric: {name}={value}")
                continue

    if not valid_metrics:
        return False

    avg_score = sum(valid_metrics) / len(valid_metrics)
    return avg_score >= threshold
async def direct_evaluate(evaluate_program_path: str,program_path: str,config:Config) -> EvaluationResult:
    """
    Directly evaluate a program using the evaluation function with timeout

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metric name to score

    Raises:
        asyncio.TimeoutError: If evaluation exceeds timeout
        Exception: If evaluation function raises an exception
    """
    #logger.info(f"开始直接评估程序: {program_path}")
    #logger.info(f"评估文件: {evaluate_program_path}")
    # Create a coroutine that runs the evaluation function in an executor
    async def run_evaluation():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _load_evaluation_function(evaluate_program_path), program_path)

    # Run the evaluation with timeout - let exceptions bubble up for retry handling
    result = await asyncio.wait_for(run_evaluation(), timeout=config.evaluator.timeout)

    # Validate result
    if not isinstance(result, dict):
        #logger.warning(f"Evaluation returned non-dictionary result: {result}")
        return EvaluationResult(metrics={"error": 0.0})

    return EvaluationResult.from_dict(result)

async def cascade_evaluate(
    program_path: str,
    evaluation_file:str,
    config:Config,
) -> EvaluationResult:
    """
    Run cascade evaluation with increasingly challenging test cases

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics or EvaluationResult with metrics and artifacts
    """
    # Import the evaluation module to get cascade functions if they exist
    #logger.info(f"开始分级评估程序: {program_path}")
    #logger.info(f"评估文件: {evaluation_file}")
    try:
        # Add the evaluation file's directory to Python path so it can import local modules
        eval_dir = os.path.dirname(os.path.abspath(evaluation_file))
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
            #logger.debug(f"Added {eval_dir} to Python path for cascade evaluation")

        spec = importlib.util.spec_from_file_location("evaluation_module", evaluation_file)
        if spec is None or spec.loader is None:
            return await direct_evaluate(program_path,evaluation_file,config)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Check if cascade functions exist
        if not hasattr(module, "evaluate_stage1"):
            return await direct_evaluate(program_path,evaluation_file,config)

        # Run first stage with timeout
        try:

            async def run_stage1():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, module.evaluate_stage1, program_path)

            stage1_result = await asyncio.wait_for(run_stage1(), timeout=config.evaluator.timeout)
            stage1_eval_result = EvaluationResult.from_dict(stage1_result)
            
        except asyncio.TimeoutError:
            #logger.warning(f"Stage 1 evaluation timed out after {config.evaluator.timeout}s")
            return EvaluationResult(
                metrics={"stage1_passed": 0.0, "error": 0.0, "timeout": True},
                artifacts={
                    "failure_stage": "stage1",
                    "timeout": True,
                },
            )
            
        except Exception as e:
            #logger.error(f"Error in stage 1 evaluation: {str(e)}")
            # Capture stage 1 failure as artifacts
            return EvaluationResult(
                metrics={"stage1_passed": 0.0, "error": 0.0},
                artifacts={
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    "failure_stage": "stage1",
                },
            )
            

        # Check threshold
        if not passes_threshold(
            stage1_eval_result.metrics, config.evaluator.cascade_thresholds[0]
        ):
            return stage1_eval_result

        # Check if second stage exists
        if not hasattr(module, "evaluate_stage2"):
            return stage1_eval_result

        # Run second stage with timeout
        try:
            async def run_stage2():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, module.evaluate_stage2, program_path)

            stage2_result = await asyncio.wait_for(run_stage2(), timeout=config.evaluator.timeout)
            stage2_eval_result = EvaluationResult.from_dict(stage2_result)
        except asyncio.TimeoutError:
            #logger.warning(f"Stage 2 evaluation timed out after {config.evaluator.timeout}s")
            # Capture stage 2 failure, but keep stage 1 results
            stage1_eval_result.artifacts.update(
                {
                    "stage2_timeout": True,
                    "failure_stage": "stage2",
                }
            )
            stage1_eval_result.metrics["stage2_passed"] = 0.0
            stage1_eval_result.metrics["timeout"] = True
            return stage1_eval_result
        except Exception as e:
            #logger.error(f"Error in stage 2 evaluation: {str(e)}")
            # Capture stage 2 failure, but keep stage 1 results
            stage1_eval_result.artifacts.update(
                {
                    "stage2_stderr": str(e),
                    "stage2_traceback": traceback.format_exc(),
                    "failure_stage": "stage2",
                }
            )
            stage1_eval_result.metrics["stage2_passed"] = 0.0
            return stage1_eval_result

        # 到这里 已经完成了stage1和stage2的评估 并且stage1的评估结果通过了阈值
        # 接下来进行stage3的评估
        # Merge results from stage 1 and 2
        merged_metrics = {}
        # Convert all values to float to avoid type errors
        for name, value in stage1_eval_result.metrics.items():
            if isinstance(value, (int, float)) and name != "error":
                merged_metrics[name] = float(value)

        for name, value in stage2_eval_result.metrics.items():
            if isinstance(value, (int, float)) and name != "error":
                merged_metrics[name] = float(value)

        # Merge artifacts
        merged_artifacts = {}
        merged_artifacts.update(stage1_eval_result.artifacts)
        merged_artifacts.update(stage2_eval_result.artifacts)

        merged_result = EvaluationResult(metrics=merged_metrics, artifacts=merged_artifacts)

        # Check threshold for stage 3
        if len(config.evaluator.cascade_thresholds) < 2 or not passes_threshold(
            merged_result.metrics, config.evaluator.cascade_thresholds[1]
        ):
            return merged_result

        # Check if third stage exists
        if not hasattr(module, "evaluate_stage3"):
            return merged_result

        # Run third stage with timeout
        try:

            async def run_stage3():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, module.evaluate_stage3, program_path)

            stage3_result = await asyncio.wait_for(run_stage3(), timeout=config.evaluator.timeout)
            stage3_eval_result = EvaluationResult.from_dict(stage3_result)
        except asyncio.TimeoutError:
            #logger.warning(f"Stage 3 evaluation timed out after {config.evaluator.timeout}s")
            # Capture stage 3 failure, but keep previous results
            merged_result.artifacts.update(
                {
                    "stage3_timeout": True,
                    "failure_stage": "stage3",
                }
            )
            merged_result.metrics["stage3_passed"] = 0.0
            merged_result.metrics["timeout"] = True
            return merged_result
        except Exception as e:
            #logger.error(f"Error in stage 3 evaluation: {str(e)}")
            # Capture stage 3 failure, but keep previous results
            merged_result.artifacts.update(
                {
                    "stage3_stderr": str(e),
                    "stage3_traceback": traceback.format_exc(),
                    "failure_stage": "stage3",
                }
            )
            merged_result.metrics["stage3_passed"] = 0.0
            return merged_result

        # Merge stage 3 results
        for name, value in stage3_eval_result.metrics.items():
            if isinstance(value, (int, float)) and name != "error":
                merged_result.metrics[name] = float(value)

        merged_result.artifacts.update(stage3_eval_result.artifacts)

        return merged_result

    except Exception as e:
        #logger.error(f"Error in cascade evaluation: {str(e)}")
        # Return proper cascade failure result instead of re-raising
        return EvaluationResult(
            metrics={"stage1_passed": 0.0, "error": 0.0},
            artifacts={
                "stderr": str(e),
                "traceback": traceback.format_exc(),
                "failure_stage": "cascade_setup",
            },
        )


if __name__ == '__main__':
    evaluate_function = _load_evaluation_function('/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/evaluator.py')
    async def test_evaluate(eval_path,init_path,config):
        result = await direct_evaluate(eval_path,init_path,config)
        print(result)
    asyncio.run(test_evaluate('/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/evaluator.py','/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/initial_program.py',Config.from_yaml('/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml')))
