

from openevolve_graph.program import Program
from openevolve_graph.Config import Config
from typing import Dict,Any,Callable
import os 
import sys 
import importlib.util
import logging
import time 
logger = logging.getLogger(__name__)
import tempfile
import asyncio
def _load_evaluation_function(evaluation_file:str) -> Callable:
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

async def direct_evaluate(evaluate_program_path: str,program_path: str,config:Config) -> Dict[str, float]:
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

    # Create a coroutine that runs the evaluation function in an executor
    async def run_evaluation():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _load_evaluation_function(evaluate_program_path), program_path)

    # Run the evaluation with timeout - let exceptions bubble up for retry handling
    result = await asyncio.wait_for(run_evaluation(), timeout=config.evaluator.timeout)

    # Validate result
    if not isinstance(result, dict):
        logger.warning(f"Evaluation returned non-dictionary result: {result}")
        return {"error": 0.0}

    return result


if __name__ == '__main__':
    evaluate_function = _load_evaluation_function('/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/evaluator.py')
    async def test_evaluate(eval_path,init_path,config):
        result = await direct_evaluate(eval_path,init_path,config)
        print(result)
    asyncio.run(test_evaluate('/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/evaluator.py','/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/initial_program.py',Config.from_yaml('/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml')))
