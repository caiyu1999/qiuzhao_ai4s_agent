

from program import Program
from openevolve_graph.Config import Config
from typing import Dict,Any
import os 
import sys 
import importlib.util
import logging
import time 
logger = logging.getLogger(__name__)
import tempfile

def _load_evaluation_function(evaluation_file:str) -> None:
        """Load the evaluation function from the evaluation file"""
        if not os.path.exists(evaluation_file):
            raise ValueError(f"Evaluation file {evaluation_file} not found")

        try:
            # Add the evaluation file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for local imports")

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
            logger.info(f"Successfully loaded evaluation function from {evaluation_file}")
            return evaluate_function
        except Exception as e:
            logger.error(f"Error loading evaluation function: {str(e)}")
            raise

async def evaluate_program(
    program_code: str,
    program_id: str = "",
    config:Config,
) -> Dict[str, float]:
    """
    Evaluate a program and return scores

    Args:
        program_code: Code to evaluate
        program_id: Optional ID for logging

    Returns:
        Dictionary of metric name to score
    """
    start_time = time.time()
    program_id_str = f" {program_id}" if program_id else ""

    # Check if artifacts are enabled
    artifacts_enabled = os.environ.get("ENABLE_ARTIFACTS", "true").lower() == "true"

    # Retry logic for evaluation
    last_exception = None
    for attempt in range(self.config.max_retries + 1):
        # Create a temporary file for the program
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(program_code.encode("utf-8"))
            temp_file_path = temp_file.name

        try:
            # Run evaluation
            if self.config.cascade_evaluation:
                # Run cascade evaluation
                result = await self._cascade_evaluate(temp_file_path)
            else:
                # Run direct evaluation
                result = await self._direct_evaluate(temp_file_path)

            # Process the result based on type
            eval_result = self._process_evaluation_result(result)

            # Check if this was a timeout and capture artifacts if enabled
            if artifacts_enabled and program_id and eval_result.metrics.get("timeout") is True:
                if program_id not in self._pending_artifacts:
                    self._pending_artifacts[program_id] = {}

                self._pending_artifacts[program_id].update(
                    {
                        "timeout": True,
                        "timeout_duration": self.config.timeout,
                        "failure_stage": "evaluation",
                        "error_type": "timeout",
                    }
                )

            # Add LLM feedback if configured
            llm_eval_result = None
            if self.config.use_llm_feedback and self.llm_ensemble:
                llm_result = await self._llm_evaluate(program_code, program_id=program_id)
                llm_eval_result = self._process_evaluation_result(llm_result)

                # Combine metrics
                for name, value in llm_result.metrics.items():
                    eval_result.metrics[f"llm_{name}"] = value * self.config.llm_feedback_weight

            # Store artifacts if enabled and present
            if (
                artifacts_enabled
                and (
                    eval_result.has_artifacts()
                    or (llm_eval_result and llm_eval_result.has_artifacts())
                )
                and program_id
            ):
                if program_id not in self._pending_artifacts:
                    self._pending_artifacts[program_id] = {}

                # Merge eval_result artifacts with llm artifacts if they exist
                if eval_result.has_artifacts():
                    self._pending_artifacts[program_id].update(eval_result.artifacts)
                    logger.debug(
                        f"Program{program_id_str} returned artifacts: "
                        f"{eval_result.artifacts}"
                    )

                if llm_eval_result and llm_eval_result.has_artifacts():
                    self._pending_artifacts[program_id].update(llm_eval_result.artifacts)
                    logger.debug(
                        f"Program{program_id_str} returned LLM artifacts: "
                        f"{llm_eval_result.artifacts}"
                    )

            elapsed = time.time() - start_time
            logger.info(
                f"Evaluated program{program_id_str} in {elapsed:.2f}s: "
                f"{format_metrics_safe(eval_result.metrics)}"
            )

            # Return just metrics for backward compatibility
            return eval_result.metrics

        except asyncio.TimeoutError:
            # Handle timeout specially - don't retry, just return timeout result
            logger.warning(f"Evaluation timed out after {self.config.timeout}s")

            # Capture timeout artifacts if enabled
            if artifacts_enabled and program_id:
                self._pending_artifacts[program_id] = {
                    "timeout": True,
                    "timeout_duration": self.config.timeout,
                    "failure_stage": "evaluation",
                    "error_type": "timeout",
                }

            return {"error": 0.0, "timeout": True}

        except Exception as e:
            last_exception = e
            logger.warning(
                f"Evaluation attempt {attempt + 1}/{self.config.max_retries + 1} failed for program{program_id_str}: {str(e)}"
            )
            traceback.print_exc()

            # Capture failure artifacts if enabled
            if artifacts_enabled and program_id:
                self._pending_artifacts[program_id] = {
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    "failure_stage": "evaluation",
                    "attempt": attempt + 1,
                }

            # If this is not the last attempt, wait a bit before retrying
            if attempt < self.config.max_retries:
                await asyncio.sleep(1.0)  # Wait 1 second before retry

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    # All retries failed
    logger.error(
        f"All evaluation attempts failed for program{program_id_str}. Last error: {str(last_exception)}"
    )
    return {"error": 0.0}

    
    
if __name__ == '__main__':
    evaluate_function = _load_evaluation_function('/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/evaluator.py')
