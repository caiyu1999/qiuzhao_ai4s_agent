

from openevolve_graph.program import Program
from openevolve_graph.Config import Config
from typing import Dict,Any,Callable
import os 
import sys 
import importlib.util
import logging
import time 

class DetailedLogger:
    """详细的日志记录器，用于精准定位评估问题"""
    
    def __init__(self, name: str = "Evaluator"):
        self.name = name
        self.logger = logging.getLogger(__name__)
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[INFO] {timestamp} | {self.name} | {message}"
        if extra_info:
            log_msg += f" | {extra_info}"
        self.logger.info(log_msg)
    
    def error(self, message: str, **kwargs):
        """记录错误日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[ERROR] {timestamp} | {self.name} | {message}"
        if extra_info:
            log_msg += f" | {extra_info}"
        self.logger.error(log_msg)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[WARNING] {timestamp} | {self.name} | {message}"
        if extra_info:
            log_msg += f" | {extra_info}"
        self.logger.warning(log_msg)
    
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[DEBUG] {timestamp} | {self.name} | {message}"
        if extra_info:
            log_msg += f" | {extra_info}"
        self.logger.debug(log_msg)
    
    def step(self, step_name: str, **kwargs):
        """记录步骤日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[STEP] {timestamp} | {self.name} | {step_name}"
        if extra_info:
            log_msg += f" | {extra_info}"
        self.logger.info(log_msg)

# Create global logger instance
logger = DetailedLogger("Evaluator")
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
        logger.step("Loading evaluation function", evaluation_file=evaluation_file)
        
        if not os.path.exists(evaluation_file):
            logger.error("Evaluation file not found", evaluation_file=evaluation_file)
            raise ValueError(f"Evaluation file {evaluation_file} not found")

        try:
            # Add the evaluation file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(evaluation_file))
            logger.debug("Evaluation directory determined", 
                        evaluation_file=evaluation_file,
                        eval_dir=eval_dir)
            
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug("Added evaluation directory to Python path", eval_dir=eval_dir)

            logger.step("Creating module spec", evaluation_file=evaluation_file)
            spec = importlib.util.spec_from_file_location("evaluation_module", evaluation_file)
            if spec is None or spec.loader is None:
                logger.error("Failed to create module spec", evaluation_file=evaluation_file)
                raise ImportError(f"Failed to load spec from {evaluation_file}")

            logger.step("Loading evaluation module", evaluation_file=evaluation_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules["evaluation_module"] = module
            spec.loader.exec_module(module)
            logger.debug("Evaluation module loaded successfully", evaluation_file=evaluation_file)

            logger.step("Checking for evaluate function", evaluation_file=evaluation_file)
            if not hasattr(module, "evaluate"):
                logger.error("Evaluate function not found in module", 
                           evaluation_file=evaluation_file,
                           available_functions=dir(module))
                raise AttributeError(
                    f"Evaluation file {evaluation_file} does not contain an 'evaluate' function"
                )

            evaluate_function = module.evaluate
            logger.info("Evaluation function loaded successfully", 
                       evaluation_file=evaluation_file,
                       function_name="evaluate")
            return evaluate_function
        except Exception as e:
            logger.error("Error loading evaluation function", 
                        evaluation_file=evaluation_file,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        traceback=traceback.format_exc())
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
    logger.step("Checking threshold", threshold=threshold, metrics_count=len(metrics) if metrics else 0)
    
    if not metrics:
        logger.debug("No metrics provided, threshold check failed")
        return False

    # Calculate average score, skipping non-numeric values and 'error' key
    valid_metrics = []
    invalid_metrics = []
    for name, value in metrics.items():
        # Skip 'error' keys and ensure values are numeric
        if name != "error" and isinstance(value, (int, float)):
            try:
                valid_metrics.append(float(value))
            except (TypeError, ValueError):
                invalid_metrics.append(f"{name}={value}")
                logger.warning("Skipping non-numeric metric", metric_name=name, metric_value=value)
                continue

    logger.debug("Threshold check metrics processed", 
                valid_metrics_count=len(valid_metrics),
                invalid_metrics_count=len(invalid_metrics),
                invalid_metrics=invalid_metrics)

    if not valid_metrics:
        logger.debug("No valid metrics found, threshold check failed")
        return False

    avg_score = sum(valid_metrics) / len(valid_metrics)
    threshold_passed = avg_score >= threshold
    
    logger.info("Threshold check completed", 
               average_score=avg_score,
               threshold=threshold,
               threshold_passed=threshold_passed,
               valid_metrics=valid_metrics)
    
    return threshold_passed
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
    logger.step("Starting direct evaluation", 
               program_path=program_path,
               evaluation_file=evaluate_program_path,
               timeout=config.evaluator.timeout)
    
    # Create a coroutine that runs the evaluation function in an executor
    async def run_evaluation():
        logger.step("Loading evaluation function for direct evaluation", evaluation_file=evaluate_program_path)
        try:
            evaluate_function = _load_evaluation_function(evaluate_program_path)
            logger.step("Running evaluation function", program_path=program_path)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, evaluate_function, program_path)
            logger.debug("Evaluation function completed", 
                        program_path=program_path,
                        result_type=type(result))
            return result
        except Exception as e:
            logger.error("Error in evaluation function execution", 
                        program_path=program_path,
                        evaluation_file=evaluate_program_path,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        traceback=traceback.format_exc())
            raise

    try:
        # Run the evaluation with timeout - let exceptions bubble up for retry handling
        logger.step("Starting evaluation with timeout", 
                   program_path=program_path,
                   timeout=config.evaluator.timeout)
        result = await asyncio.wait_for(run_evaluation(), timeout=config.evaluator.timeout)
        logger.info("Direct evaluation completed successfully", 
                   program_path=program_path,
                   result_type=type(result))

        # Validate result
        if not isinstance(result, dict):
            logger.warning("Evaluation returned non-dictionary result", 
                          program_path=program_path,
                          result_type=type(result),
                          result_value=str(result)[:200])  # Truncate long results
            return EvaluationResult(metrics={"error": 0.0})

        logger.info("Direct evaluation result validated", 
                   program_path=program_path,
                   result_keys=list(result.keys()) if isinstance(result, dict) else [])
        return EvaluationResult.from_dict(result)
        
    except asyncio.TimeoutError:
        logger.error("Direct evaluation timed out", 
                    program_path=program_path,
                    timeout=config.evaluator.timeout)
        raise
    except Exception as e:
        logger.error("Direct evaluation failed", 
                    program_path=program_path,
                    evaluation_file=evaluate_program_path,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc())
        raise

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
    logger.step("Starting cascade evaluation", 
               program_path=program_path,
               evaluation_file=evaluation_file,
               timeout=config.evaluator.timeout)
    
    # Import the evaluation module to get cascade functions if they exist
    try:
        logger.step("Setting up cascade evaluation environment", evaluation_file=evaluation_file)
        
        # Add the evaluation file's directory to Python path so it can import local modules
        eval_dir = os.path.dirname(os.path.abspath(evaluation_file))
        logger.debug("Evaluation directory determined", eval_dir=eval_dir)
        
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
            logger.debug("Added evaluation directory to Python path for cascade evaluation", eval_dir=eval_dir)

        logger.step("Loading evaluation module for cascade", evaluation_file=evaluation_file)
        spec = importlib.util.spec_from_file_location("evaluation_module", evaluation_file)
        if spec is None or spec.loader is None:
            logger.warning("Failed to create module spec, falling back to direct evaluation", 
                          evaluation_file=evaluation_file)
            return await direct_evaluate(program_path,evaluation_file,config)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.debug("Evaluation module loaded for cascade", evaluation_file=evaluation_file)

        # Check if cascade functions exist
        logger.step("Checking for cascade evaluation functions", evaluation_file=evaluation_file)
        available_functions = [attr for attr in dir(module) if attr.startswith('evaluate_stage')]
        logger.debug("Available cascade functions", 
                    evaluation_file=evaluation_file,
                    available_functions=available_functions)
        
        if not hasattr(module, "evaluate_stage1"):
            logger.warning("evaluate_stage1 function not found, falling back to direct evaluation", 
                          evaluation_file=evaluation_file,
                          available_functions=available_functions)
            return await direct_evaluate(program_path,evaluation_file,config)

        # Run first stage with timeout
        logger.step("Starting stage 1 evaluation", 
                   program_path=program_path,
                   timeout=config.evaluator.timeout)
        
        try:
            async def run_stage1():
                logger.step("Running stage 1 evaluation function", program_path=program_path)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, module.evaluate_stage1, program_path)
                logger.debug("Stage 1 evaluation function completed", 
                            program_path=program_path,
                            result_type=type(result))
                return result

            stage1_result = await asyncio.wait_for(run_stage1(), timeout=config.evaluator.timeout)
            stage1_eval_result = EvaluationResult.from_dict(stage1_result)
            logger.info("Stage 1 evaluation completed successfully", 
                       program_path=program_path,
                       result_keys=list(stage1_result.keys()) if isinstance(stage1_result, dict) else [])
            
        except asyncio.TimeoutError:
            logger.error("Stage 1 evaluation timed out", 
                        program_path=program_path,
                        timeout=config.evaluator.timeout)
            return EvaluationResult(
                metrics={"stage1_passed": 0.0, "error": 0.0, "timeout": True},
                artifacts={
                    "failure_stage": "stage1",
                    "timeout": True,
                },
            )
            
        except Exception as e:
            logger.error("Error in stage 1 evaluation", 
                        program_path=program_path,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        traceback=traceback.format_exc())
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
        logger.step("Checking stage 1 threshold", 
                   program_path=program_path,
                   threshold=config.evaluator.cascade_thresholds[0])
        
        if not passes_threshold(
            stage1_eval_result.metrics, config.evaluator.cascade_thresholds[0]
        ):
            logger.info("Stage 1 threshold not passed, stopping cascade evaluation", 
                       program_path=program_path,
                       threshold=config.evaluator.cascade_thresholds[0])
            return stage1_eval_result

        # Check if second stage exists
        logger.step("Checking for stage 2 evaluation function", evaluation_file=evaluation_file)
        if not hasattr(module, "evaluate_stage2"):
            logger.info("evaluate_stage2 function not found, returning stage 1 results", 
                       evaluation_file=evaluation_file)
            return stage1_eval_result

        # Run second stage with timeout
        logger.step("Starting stage 2 evaluation", 
                   program_path=program_path,
                   timeout=config.evaluator.timeout)
        
        try:
            async def run_stage2():
                logger.step("Running stage 2 evaluation function", program_path=program_path)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, module.evaluate_stage2, program_path)
                logger.debug("Stage 2 evaluation function completed", 
                            program_path=program_path,
                            result_type=type(result))
                return result

            stage2_result = await asyncio.wait_for(run_stage2(), timeout=config.evaluator.timeout)
            stage2_eval_result = EvaluationResult.from_dict(stage2_result)
            logger.info("Stage 2 evaluation completed successfully", 
                       program_path=program_path,
                       result_keys=list(stage2_result.keys()) if isinstance(stage2_result, dict) else [])
        except asyncio.TimeoutError:
            logger.error("Stage 2 evaluation timed out", 
                        program_path=program_path,
                        timeout=config.evaluator.timeout)
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
            logger.error("Error in stage 2 evaluation", 
                        program_path=program_path,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        traceback=traceback.format_exc())
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
        logger.step("Merging stage 1 and 2 results", program_path=program_path)
        
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
        logger.debug("Stage 1 and 2 results merged", 
                    program_path=program_path,
                    merged_metrics_count=len(merged_metrics),
                    merged_artifacts_count=len(merged_artifacts))

        # Check threshold for stage 3
        logger.step("Checking stage 2 threshold for stage 3", 
                   program_path=program_path,
                   threshold=config.evaluator.cascade_thresholds[1] if len(config.evaluator.cascade_thresholds) >= 2 else None)
        
        if len(config.evaluator.cascade_thresholds) < 2 or not passes_threshold(
            merged_result.metrics, config.evaluator.cascade_thresholds[1]
        ):
            logger.info("Stage 2 threshold not passed, stopping cascade evaluation", 
                       program_path=program_path,
                       threshold=config.evaluator.cascade_thresholds[1] if len(config.evaluator.cascade_thresholds) >= 2 else None)
            return merged_result

        # Check if third stage exists
        logger.step("Checking for stage 3 evaluation function", evaluation_file=evaluation_file)
        if not hasattr(module, "evaluate_stage3"):
            logger.info("evaluate_stage3 function not found, returning merged results", 
                       evaluation_file=evaluation_file)
            return merged_result

        # Run third stage with timeout
        logger.step("Starting stage 3 evaluation", 
                   program_path=program_path,
                   timeout=config.evaluator.timeout)
        
        try:
            async def run_stage3():
                logger.step("Running stage 3 evaluation function", program_path=program_path)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, module.evaluate_stage3, program_path)
                logger.debug("Stage 3 evaluation function completed", 
                            program_path=program_path,
                            result_type=type(result))
                return result

            stage3_result = await asyncio.wait_for(run_stage3(), timeout=config.evaluator.timeout)
            stage3_eval_result = EvaluationResult.from_dict(stage3_result)
            logger.info("Stage 3 evaluation completed successfully", 
                       program_path=program_path,
                       result_keys=list(stage3_result.keys()) if isinstance(stage3_result, dict) else [])
        except asyncio.TimeoutError:
            logger.error("Stage 3 evaluation timed out", 
                        program_path=program_path,
                        timeout=config.evaluator.timeout)
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
            logger.error("Error in stage 3 evaluation", 
                        program_path=program_path,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        traceback=traceback.format_exc())
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
        logger.step("Merging stage 3 results", program_path=program_path)
        for name, value in stage3_eval_result.metrics.items():
            if isinstance(value, (int, float)) and name != "error":
                merged_result.metrics[name] = float(value)

        merged_result.artifacts.update(stage3_eval_result.artifacts)
        logger.info("All cascade stages completed successfully", 
                   program_path=program_path,
                   final_metrics_count=len(merged_result.metrics),
                   final_artifacts_count=len(merged_result.artifacts))

        return merged_result

    except Exception as e:
        logger.error("Error in cascade evaluation setup", 
                    program_path=program_path,
                    evaluation_file=evaluation_file,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc())
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
