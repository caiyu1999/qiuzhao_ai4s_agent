import os
import json
from typing import Dict
import uuid
import tempfile
import traceback
from pydantic import BaseModel
# from openevolve_graph.Graph.Graph_Node_ABC import NodeType
from openevolve_graph.Graph.Graph_Node_ABC import NodeResult,SyncNode,AsyncNode
from openevolve_graph.Graph.Graph_state import GraphState, IslandStatus ,IslandState
from openevolve_graph.Config import Config
from openevolve_graph.program import Program
from openevolve_graph.visualization.socket_sc import SimpleClient
from openevolve_graph.utils.utils import (
    parse_full_rewrite,
    safe_numeric_average,
    _calculate_feature_coords,
    _feature_coords_to_key,
    format_metrics_safe,
    _get_artifact_size,
    _artifact_serializer,
    store_artifacts,
    get_artifacts,
    get_top_programs,
    _is_better
    
    )
from openevolve_graph.Prompt.sampler import PromptSampler_langchain
from openevolve_graph.utils.utils import (
    load_initial_program,
    extract_code_language,
    apply_diff,extract_diffs,
    format_diff_summary,
    parse_full_rewrite
    )
from openevolve_graph.utils.thread_safe_programs import Programs_container
from openevolve_graph.models.structed_output import (
    ResponseFormatter_template_diff,
    ResponseFormatter_template_rewrite,
    ResponseFormatter_template_evaluator
    )
from evaluator import (
    direct_evaluate,
    cascade_evaluate,
    EvaluationResult,
    passes_threshold,
)
from openevolve_graph.models.LLMs import LLMs,LLMs_evalutor
from typing import Optional, Union, Type, List, Any, Dict, Any, Tuple
import re
import random
from typing_extensions import TypedDict
import time
import asyncio
import os
import logging



# 详细的日志类
class DetailedLogger:
    """详细的日志记录器，用于精准定位问题"""
    
    def __init__(self, name: str = "GraphNode"):
        self.name = name
        self.logger = logging.getLogger(__name__)
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[INFO] {timestamp} | {self.name} | {message}"
        if extra_info:
            log_msg += f" | {extra_info}"
        # 这里会对接上层的父代logger
        self.logger.info(log_msg)
    
    def error(self, message: str, **kwargs):
        """记录错误日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[ERROR] {timestamp} | {self.name} | {message}"
        if extra_info:
            log_msg += f" | {extra_info}"
        # 这里会对接上层的父代logger
        self.logger.error(log_msg)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[WARNING] {timestamp} | {self.name} | {message}"
        if extra_info:
            log_msg += f" | {extra_info}"
        # 这里会对接上层的父代logger
        self.logger.warning(log_msg)
    
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[DEBUG] {timestamp} | {self.name} | {message}"
        if extra_info:
            log_msg += f" | {extra_info}"
        # 这里会对接上层的父代logger
        self.logger.debug(log_msg)
    
    def step(self, step_name: str, **kwargs):
        """记录步骤日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
        log_msg = f"[STEP] {timestamp} | {self.name} | {step_name}"
        if extra_info:
            log_msg += f" | {extra_info}"
        # 这里会对接上层的父代logger
        self.logger.info(log_msg)

# 创建全局日志实例
logger = DetailedLogger("GraphNode")

class node_init_status(AsyncNode):
    '''
    初始化图的状态
    
    功能描述：
    - 验证配置参数的有效性（程序路径、评估器路径、岛屿数量等）
    - 加载并评估初始程序，获取初始指标
    - 创建初始Program对象，包含代码、语言、指标等信息
    - 初始化所有岛屿的状态，为每个岛屿分配初始程序
    - 设置全局数据结构（程序库、归档、特征映射等）
    
    更新的状态字段：
    - init_program: 初始程序ID
    - best_program: 最佳程序对象
    - best_program_id: 最佳程序ID
    - best_metrics: 最佳程序指标
    - num_islands: 岛屿数量
    - archive: 精英程序归档
    - all_programs: 全局程序库
    - evaluation_program: 评估器路径
    - language: 编程语言
    - file_extension: 文件扩展名
    - islands_id: 岛屿ID列表
    - feature_map: 特征映射字典
    - islands: 岛屿状态字典
    
    输入要求：
    - GraphState: 空的图状态对象
    
    输出结果：
    - Dict: 包含初始化后的所有状态字段
    '''

    def __init__(self,config:Config,next_meeting:int):
        self.config = config
        self.num_islands = config.island.num_islands
        self.prompt_sampler = PromptSampler_langchain(config=config.prompt)
        self.llm_evaluator = LLMs_evalutor(config=config)
        if self.config.evaluator.use_llm_feedback: #如果是基于差异的演化 那么使用
            self.structure = ResponseFormatter_template_evaluator
            self.key = ["readability","maintainability","efficiency","other_information"]
        else:
            self.structure = None
            self.key = None
        self.next_meeting = next_meeting

    async def execute_async(self, state: GraphState) -> NodeResult | Dict[str, Any]:
        config = self.config
        
        logger.step("Starting node_init_status execution", node_type="node_init_status")
        
        # 验证必要的配置参数
        logger.info("Validating configuration parameters", 
                    init_program_path=config.init_program_path,
                    evaluator_file_path=config.evalutor_file_path,
                    num_islands=config.island.num_islands)
        
        if config.init_program_path == "":
            logger.error("Configuration validation failed: init_program_path is empty")
            raise ValueError("init_program is not set")
        if config.evalutor_file_path == "":
            logger.error("Configuration validation failed: evaluator_file_path is empty")
            raise ValueError("evaluator_file_path is not set")
        if config.island.num_islands <= 0:
            logger.error("Configuration validation failed: num_islands must be greater than 0", 
                        num_islands=config.island.num_islands)
            raise ValueError("num_islands must be greater than 0")

        logger.info("Configuration validation passed successfully")

        # 提取文件信息
        file_extension = os.path.splitext(config.init_program_path)[1]
        if not file_extension:
            file_extension = ".py"  # 默认扩展名
        logger.debug("File extension extracted", file_extension=file_extension)

        # 加载和处理初始程序
        logger.step("Loading initial program", file_path=config.init_program_path)
        try:
            code = load_initial_program(config.init_program_path)
            logger.info("Initial program loaded successfully", 
                       file_path=config.init_program_path,
                       code_length=len(code))
        except Exception as e:
            logger.error("Failed to load initial program", 
                        file_path=config.init_program_path,
                        error=str(e))
            raise
        
        language = extract_code_language(code)
        logger.debug("Code language detected", language=language)

        # 生成唯一ID
        id = str(uuid.uuid4())
        logger.debug("Generated unique program ID", program_id=id)
        
        # 对初始程序进行评估 
        logger.step("Starting initial program evaluation", 
                   program_id=id,
                   evaluation_file=self.config.evalutor_file_path)
        
        init_program_code = code
        init_program_id = id
        evaluation_file = self.config.evalutor_file_path
        
        try:
            eval_result, artifact_update = await self._evaluate_program(
                init_program_code, init_program_id, evaluation_file, state
            )
            logger.debug("Initial program evaluation completed successfully",
                       program_id=init_program_id,
                       metrics=eval_result.metrics,
                       has_artifacts=bool(artifact_update))
        except Exception as e:
            logger.error("Initial program evaluation failed",
                        program_id=init_program_id,
                        error=str(e))
            raise
        
        artifact = None 
        artifacts_json = None
        artifact_dir = None
        if self.config.enable_artifacts: #如果开启工件 先生成工件 并存储工件
            logger.step("Processing artifacts", artifacts_enabled=True)
            artifact = eval_result.artifacts
            artifact.update(artifact_update) # 更新工件
            artifacts_json, artifact_dir = store_artifacts(init_program_id, artifact, state, self.config)
            logger.debug("Artifacts stored successfully",
                       program_id=init_program_id,
                       artifact_dir=artifact_dir)
        
        # 生成program对象 
        logger.step("Creating initial Program object", program_id=init_program_id)
        init_program = Program(
                id=init_program_id,
                code=init_program_code,
                language=language,
                parent_id=None,
                generation=0,
                timestamp=time.time(),
                metrics=eval_result.metrics,
                artifacts_json=artifacts_json,
                artifact_dir=artifact_dir,
                )
        logger.info("Initial Program object created successfully",
                   program_id=init_program_id,
                   language=language,
                   generation=0)
        
        # 初始化岛屿相关数据结构
        logger.step("Initializing island data structures", num_islands=config.island.num_islands)
        island_id_list = [f"{i}" for i in range(config.island.num_islands)]
        islandstate_dict = {}
        
        for island_id in island_id_list:
            logger.debug("Initializing island", island_id=island_id)
            temp_Island_state = IslandState(id=island_id)
            temp_Island_state.programs.add_program(init_program.id, init_program)
            temp_Island_state.latest_program = init_program 
            temp_Island_state.status = IslandStatus.INIT_STATE
            temp_Island_state.all_programs.add_program(init_program.id, init_program)
            temp_Island_state.archive.add_program(init_program.id, init_program)
            temp_Island_state.language = language
            temp_Island_state.all_best_program = init_program
            temp_Island_state.next_meeting = self.next_meeting
            temp_Island_state.now_meeting = 0
            islandstate_dict[island_id] = temp_Island_state
            logger.debug("Island initialized successfully", 
                        island_id=island_id,
                        next_meeting=self.next_meeting)

        logger.info("All islands initialized successfully", 
                   total_islands=len(islandstate_dict))
            
        island_programs_ = Programs_container()
        island_programs_.add_program(init_program_id, init_program)
        all_programs = island_programs_.copy()
        archive = island_programs_.copy()
        feature_map = {} # 全部程序中的特征坐标 全局更新 e.g{"program_id":[0,1,2,-1]}
        num_islands = config.island.num_islands
        islands_id = island_id_list # 岛屿的id 全局唯一 不会被更新 安全
        best_program = init_program 
        
        logger.step("node_init_status execution completed successfully",
                   program_id=init_program_id,
                   num_islands=num_islands,
                   language=language)
        
        return {
            "init_program":init_program,
            "best_program":best_program,
            "num_islands":num_islands,
            "archive":archive,
            "all_programs":all_programs,
            "evaluation_program":config.evalutor_file_path,
            "language":language,
            "file_extension":file_extension,
            "islands_id":islands_id,
            "best_program":best_program,
            "feature_map":feature_map,
            "islands":islandstate_dict,
            }
        
    def __call__(self,state:GraphState):
        logger.step("node_init_status __call__ method invoked")
        try:
            result = asyncio.run(self.execute_async(state))
            logger.info("node_init_status __call__ method completed successfully")
            return result
        except Exception as e:
            logger.error("node_init_status __call__ method failed", error=str(e))
            raise
    
    async def _llm_evaluate(self, program_code: str) -> Dict[str, Any]:
        """
        Use LLM to evaluate code quality

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging

        Returns:
            Dictionary of metric name to score
        """
        logger.step("Starting LLM evaluation", code_length=len(program_code))
        
        if not self.llm_evaluator:
            logger.warning("LLM evaluator not configured, skipping LLM evaluation")
            return {}

        try:
            # Create prompt for LLMThreadSafePrograms
            logger.debug("Building LLM evaluation prompt")
            prompt = self.prompt_sampler.build_prompt(
                current_program=program_code, template_key="evaluation"
            )
            logger.debug("LLM evaluation prompt built successfully", prompt_length=len(prompt))

            # Get LLM response
            logger.step("Invoking LLM for evaluation")
            responses = await self.llm_evaluator.invoke_parallel(
                prompt=prompt,
                structure=self.structure,
                key=self.key,
            )
            logger.info("LLM evaluation response received", 
                       response_keys=list(responses.keys()) if responses else [])

            try: # 当前版本返回的一般是一个dict {key:[value1,value2,value3]}

                artifacts = {}
                metrics = {}
                for key, value_list in responses.items():
                    if key == "other_information": # 其他信息 直接赋给artifacts
                        continue
                    length_ = len(value_list) #获取value_list的长度
                    metrics[key] = sum(value_list) / length_
                    
                artifacts['other_information'] = responses['other_information']

                logger.info("LLM evaluation metrics calculated successfully",
                           metrics=metrics,
                           artifacts_count=len(artifacts))

                return EvaluationResult(
                    metrics=metrics,
                    artifacts=artifacts,
                ).to_dict()

            except Exception as e:
                logger.error("Failed to parse LLM evaluation response", 
                           error=str(e),
                           response_type=type(responses))
                return {}

        except Exception as e:
            logger.error("LLM evaluation failed", error=str(e))
            return {}
    def _process_evaluation_result(self, result: Any) -> EvaluationResult:
        """
        Process evaluation result to handle both dict and EvaluationResult returns

        Args:
            result: Raw result from evaluation function

        Returns:
            EvaluationResult instance
        """
        logger.debug("Processing evaluation result", result_type=type(result))
        
        if isinstance(result, dict):
            # Backward compatibility - wrap dict in EvaluationResult
            logger.debug("Converting dict result to EvaluationResult")
            return EvaluationResult.from_dict(result)
        elif isinstance(result, EvaluationResult):
            # New format - use directly
            logger.debug("Using EvaluationResult directly")
            return result
        else:
            # Error case - return error metrics
            logger.warning("Unexpected evaluation result type, returning error metrics",
                         result_type=type(result))
            return EvaluationResult(metrics={"error": 0.0})

    async def _evaluate_program(
        self,
        program_code:str,
        program_id:str,
        evaluation_file:str,
        state:GraphState,
    ) -> Tuple[EvaluationResult, Dict[str, Any]]:
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

        logger.step("Starting program evaluation", 
                   program_id=program_id,
                   evaluation_file=evaluation_file,
                   code_length=len(program_code))

        # Check if artifacts are enabled 是否开启工件通道
        artifacts_enabled = self.config.enable_artifacts
        logger.debug("Artifacts configuration", artifacts_enabled=artifacts_enabled)

        # Retry logic for evaluation
        last_exception = None

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(program_code.encode("utf-8"))
            temp_file_path = temp_file.name
            logger.debug("Temporary file created", temp_file_path=temp_file_path)
            
        eval_result = EvaluationResult()
        artifact_update = {}
        
        try:
            # Run evaluation
            logger.step("Executing evaluation", 
                       evaluation_type="cascade" if self.config.evaluator.cascade_evaluation else "direct")
            
            if self.config.evaluator.cascade_evaluation:# 分级评估 会返回一个metrics和artifacts的EvaluationResult 工件在这里产生
                result = await cascade_evaluate(temp_file_path,evaluation_file,self.config)
            else:
                result = await direct_evaluate(temp_file_path,evaluation_file,self.config)

            logger.info("Evaluation execution completed", 
                       program_id=program_id,
                       result_type=type(result))

            # Process the result based on type 
            eval_result = self._process_evaluation_result(result)
            logger.info("Evaluation result processed", 
                       program_id=program_id,
                       metrics_count=len(eval_result.metrics))

            # Check if this was a timeout and capture artifacts if enabled 
            if artifacts_enabled and program_id and eval_result.metrics.get("timeout") is True:
                logger.warning("Evaluation timeout detected", 
                             program_id=program_id,
                             timeout_duration=self.config.evaluator.timeout)
                if state.all_programs.get_program(program_id).artifacts_json is not None:
                    artifact_update = {}

                artifact_update={ # 注意这里是要更新的 而不是新建
                        "timeout": True,
                        "timeout_duration": self.config.evaluator.timeout,
                        "failure_stage": "evaluation",
                        "error_type": "timeout",
                    }

            # Add LLM feedback if configured
            llm_eval_result = None
            if self.config.evaluator.use_llm_feedback and self.llm_evaluator:
                logger.step("Adding LLM feedback to evaluation", program_id=program_id)
                llm_result = await self._llm_evaluate(program_code) # 返回一个dict  {'readability': 0.85, 'maintainability': 0.8, 'efficiency': 0.7, 'other_information': ''}
                llm_eval_result = self._process_evaluation_result(llm_result)

                for name, value in llm_result.items():
                    eval_result.metrics[f"llm_{name}"] = value
                
                logger.info("LLM feedback added to evaluation", 
                           program_id=program_id,
                           llm_metrics_count=len([k for k in eval_result.metrics.keys() if k.startswith('llm_')]))

            # Store artifacts if enabled and present
            if (
                artifacts_enabled
                and (
                    eval_result.has_artifacts()
                    or (llm_eval_result and llm_eval_result.has_artifacts())
                )
                and program_id
            ):
                logger.step("Processing evaluation artifacts", program_id=program_id)
                if state.all_programs.get_program(program_id).artifacts_json is not None:
                    artifact_update = {}

                # Merge eval_result artifacts with llm artifacts if they exist
                if eval_result.has_artifacts():
                    artifact_update=eval_result.artifacts
                    logger.debug("Evaluation artifacts merged", 
                               program_id=program_id,
                               artifacts_count=len(eval_result.artifacts))

                if llm_eval_result and llm_eval_result.has_artifacts():
                    artifact_update=llm_eval_result.artifacts
                    logger.debug("LLM artifacts merged", 
                               program_id=program_id,
                               artifacts_count=len(llm_eval_result.artifacts))

            elapsed = time.time() - start_time
            logger.info("Program evaluation completed successfully", 
                       program_id=program_id,
                       elapsed_time=f"{elapsed:.2f}s",
                       final_metrics_count=len(eval_result.metrics))
            
            # Return just metrics for backward compatibility
            return eval_result , artifact_update

        except asyncio.TimeoutError: #如果是超时错误 
            # Handle timeout specially - don't retry, just return timeout result
            logger.error("Evaluation timeout occurred", 
                        program_id=program_id,
                        timeout_duration=self.config.evaluator.timeout)

            # Capture timeout artifacts if enabled
            if artifacts_enabled and program_id:
                artifact_update = {
                    "timeout": True,
                    "timeout_duration": self.config.evaluator.timeout,
                    "failure_stage": "evaluation",
                    "error_type": "timeout",
                }
                eval_result.metrics.update({"error": 0.0,"timeout": True})

            return eval_result , artifact_update

        except Exception as e:
            last_exception = e
            logger.error("Evaluation failed with exception", 
                        program_id=program_id,
                        error=str(e),
                        error_type=type(e).__name__)

            # Capture failure artifacts if enabled
            if artifacts_enabled and program_id:
                artifact_update = {
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    "failure_stage": "evaluation",
                }

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.debug("Temporary file cleaned up", temp_file_path=temp_file_path)

        # All retries failed
        logger.error("All evaluation attempts failed", 
                    program_id=program_id,
                    last_error=str(last_exception) if last_exception else "Unknown")
        eval_result.metrics.update({"error": 0.0})
        return eval_result , artifact_update
    
class node_evaluate(AsyncNode):
    '''
    程序评估节点
    
    功能描述：
    - 评估岛屿上的最新程序（latest_program）的性能指标
    - 使用配置的评估器（直接评估或分级评估）对程序进行测试
    - 可选启用LLM反馈评估，获取代码质量指标
    - 处理评估超时和异常情况，记录相关工件信息
    - 创建包含评估结果的新Program对象
    
    更新的状态字段：
    - status: 节点状态更新为EVALUATE_CHILD
    - latest_program: 包含评估结果的新Program对象
    - evaluate_success: 评估是否成功的布尔值
    - iteration: 如果评估失败，迭代次数+1
    - now_meeting: 如果评估失败，会议进度+1
    - next_meeting: 如果评估失败，下次会议时间-1
    
    输入要求：
    - IslandState: 包含latest_program的岛屿状态
    
    输出结果：
    - Dict: 包含评估状态和结果的字典
    
    异常处理：
    - 评估失败时返回None，触发重新采样机制
    - 超时情况下记录超时工件信息
    - 异常情况下记录错误堆栈信息
    '''
    def __init__(self,config:Config):
        self.config = config
        self.llm_evalutor = LLMs_evalutor(config)
        self.prompt_sampler = PromptSampler_langchain(config=config.prompt)
        self.client = None
        if self.config.evaluator.use_llm_feedback: #如果是llm反馈评估 那么使用
            self.structure = ResponseFormatter_template_evaluator
            self.key = ["readability","maintainability","efficiency","other_information"]
        else:
            self.structure = None
            self.key = None

    async def execute_async(self,state:IslandState):
        logger.step("Starting node_evaluate execution", island_id=state.id)
        
        #岛屿的并行评估
        try:
            current_program = state.latest_program.code
            current_program_id = state.latest_program.id
            evaluation_file = self.config.evalutor_file_path
            
            logger.debug("Program evaluation parameters prepared", 
                        island_id=state.id,
                        program_id=current_program_id,
                        evaluation_file=evaluation_file,
                        code_length=len(current_program))

            eval_result,artifact_update = await self._evaluate_program(current_program,current_program_id,evaluation_file,state)
            logger.info("Program evaluation completed", 
                       island_id=state.id,
                       program_id=current_program_id,
                       metrics=eval_result.metrics)
            
            changes_summary = state.change_summary
            parent_id = state.sample_program.id
            parent_metrics = state.all_programs.get_program(parent_id).metrics
            
            logger.debug("Parent program information retrieved", 
                        island_id=state.id,
                        parent_id=parent_id,
                        changes_summary=changes_summary)
            
            artifact = None 
            artifacts_json = None
            artifact_dir = None
            if self.config.enable_artifacts: #如果开启工件 先生成工件 并存储工件
                logger.step("Processing artifacts", 
                           island_id=state.id,
                           program_id=current_program_id,
                           artifacts_enabled=True)
                artifact = eval_result.artifacts
                artifact.update(artifact_update) # 更新工件
                artifacts_json,artifact_dir = store_artifacts(current_program_id,artifact,state,self.config)
                logger.info("Artifacts stored successfully", 
                           island_id=state.id,
                           program_id=current_program_id,
                           artifact_dir=artifact_dir)
            
            logger.step("Creating evaluated Program object", 
                       island_id=state.id,
                       program_id=current_program_id)
            
            current_program = Program(
                    id=current_program_id,
                    code=current_program,
                    island_id=state.id,
                    language=state.language,
                    parent_id=state.sample_program.id,
                    generation=state.iteration,
                    timestamp=time.time(),
                    metrics=eval_result.metrics,
                    metadata={
                        "changes": changes_summary,
                        "parent_metrics": parent_metrics,
                    },
                    artifacts_json=artifacts_json,
                    artifact_dir=artifact_dir,
                    )
            
            logger.info("Evaluated Program object created successfully", 
                       island_id=state.id,
                       program_id=current_program_id,
                       generation=state.iteration)
            
            return current_program
        except Exception as e:
            logger.error("Program evaluation failed with exception", 
                        island_id=state.id,
                        error=str(e),
                        error_type=type(e).__name__)
            return None
    def __call__(self,state:IslandState):
        '''
        这个节点计算初始prgram的信息 并更新current_program
        '''
        logger.step("node_evaluate __call__ method invoked", island_id=state.id)
        try:
            current_program = asyncio.run(self.execute_async(state))
            logger.info("node_evaluate __call__ method completed successfully", 
                       island_id=state.id,
                       program_created=current_program is not None)
            
            if self.client is None:
                self.client = SimpleClient(self.config.port)
            
            update_dict = {}
            if current_program is None:
                logger.error("Program evaluation failed, updating state for retry", 
                           island_id=state.id)
                #目前暂时进行下一轮
                update_dict = {
                    "status":IslandStatus.SAMPLE,
                    "evaluate_success":False,
                    "now_meeting":state.now_meeting+1,
                    "next_meeting":state.next_meeting-1,
                    "iteration":state.iteration+1,
                }
                self.client.send_message({
                    "island_id":state.id,
                    "update_dict":update_dict,
                })
                return update_dict
            
            update_dict = {
                "status":IslandStatus.UPDATE,
                "latest_program":current_program,
                "evaluate_success":True,
            }
            
            self.client.send_message({
                    "island_id":state.id,
                    "update_dict":update_dict,
                    
                })
            logger.info("node_evaluate __call__ method completed successfully",island_id=state.id)
            return update_dict
        except Exception as e:
            logger.error("node_evaluate __call__ method failed", 
                        island_id=state.id,
                        error=str(e))
            raise
        
        
    async def _llm_evaluate(self, program_code: str) -> Dict[str, Any]:
        """
        Use LLM to evaluate code quality

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging

        Returns:
            Dictionary of metric name to score
        """
        logger.step("Starting LLM evaluation", code_length=len(program_code))
        
        if not self.llm_evalutor:
            logger.warning("LLM evaluator not configured, skipping LLM evaluation")
            return {}

        try:
            # Create prompt for LLMThreadSafePrograms
            logger.debug("Building LLM evaluation prompt")
            prompt = self.prompt_sampler.build_prompt(
                current_program=program_code, template_key="evaluation"
            )
            logger.debug("LLM evaluation prompt built successfully", prompt_length=len(prompt))

            # Get LLM response
            logger.step("Invoking LLM for evaluation")
            responses = await self.llm_evalutor.invoke_parallel(
                prompt=prompt,
                structure=self.structure,
                key=self.key,
            )
            logger.info("LLM evaluation response received", 
                       response_keys=list(responses.keys()) if responses else [])

            try: # 当前版本返回的一般是一个dict {key:[value1,value2,value3]}

                artifacts = {}
                metrics = {}
                # {'readability': [0.85], 'maintainability': [0.75], 'efficiency': [0.8], 'reasoning': ['The code is well-structured and uses clear function separation, making it understandable and maintainable. The use of numpy for array operations enhances performance, but the nested loops in the `compute_max_radii`
                # function could be optimized further. Overall, the constructor approach is explicit and avoids the complexity of iterative algorithms.']}
                for key, value_list in responses.items():
                    if key =="other_information": # 其他信息 直接赋给artifacts
                        continue
                    length_ = len(value_list) #获取value_list的长度
                    metrics[key] = sum(value_list) / length_
                
                artifacts['other_information'] = responses['other_information']
                logger.info("LLM evaluation metrics calculated successfully",
                           metrics=metrics,
                           artifacts_count=len(artifacts))
                
                return EvaluationResult(
                    metrics=metrics,
                    artifacts=artifacts,
                ).to_dict()

            except Exception as e:
                logger.error("Failed to parse LLM evaluation response", 
                           error=str(e),
                           response_type=type(responses))
                return {}

        except Exception as e:
            logger.error("LLM evaluation failed", error=str(e))
            traceback.print_exc()
            return {}
    def _process_evaluation_result(self, result: Any) -> EvaluationResult:
        """
        Process evaluation result to handle both dict and EvaluationResult returns

        Args:
            result: Raw result from evaluation function

        Returns:
            EvaluationResult instance
        """
        logger.debug("Processing evaluation result", result_type=type(result))
        
        if isinstance(result, dict):
            # Backward compatibility - wrap dict in EvaluationResult
            logger.debug("Converting dict result to EvaluationResult")
            return EvaluationResult.from_dict(result)
        elif isinstance(result, EvaluationResult):
            # New format - use directly
            logger.debug("Using EvaluationResult directly")
            return result
        else:
            # Error case - return error metrics
            logger.warning("Unexpected evaluation result type, returning error metrics",
                         result_type=type(result))
            return EvaluationResult(metrics={"error": 0.0})

    async def _evaluate_program(
        self,
        program_code,
        program_id,
        evaluation_file,
        state:IslandState,
    ) -> Tuple[EvaluationResult, Dict[str, Any]]:
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

        logger.step("Starting program evaluation", 
                   program_id=program_id,
                   evaluation_file=evaluation_file,
                   code_length=len(program_code))

        # Check if artifacts are enabled 是否开启工件通道
        artifacts_enabled = self.config.enable_artifacts
        logger.debug("Artifacts configuration", artifacts_enabled=artifacts_enabled)

        # Retry logic for evaluation
        last_exception = None

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(program_code.encode("utf-8"))
            temp_file_path = temp_file.name
            logger.debug("Temporary file created", temp_file_path=temp_file_path)
            
        eval_result = EvaluationResult()
        artifact_update = {}
        
        try:
            # Run evaluation
            logger.step("Executing evaluation", 
                       evaluation_type="cascade" if self.config.evaluator.cascade_evaluation else "direct")
            
            if self.config.evaluator.cascade_evaluation:# 分级评估 会返回一个metrics和artifacts的EvaluationResult 工件在这里产生
                result = await cascade_evaluate(temp_file_path,evaluation_file,self.config)
            else:
                result = await direct_evaluate(temp_file_path,evaluation_file,self.config)

            logger.info("Evaluation execution completed", 
                       program_id=program_id,
                       result_type=type(result))

            # Process the result based on type 
            eval_result = self._process_evaluation_result(result)
            logger.info("Evaluation result processed", 
                       program_id=program_id,
                       metrics_count=len(eval_result.metrics))

            # Check if this was a timeout and capture artifacts if enabled 
            if artifacts_enabled and program_id and eval_result.metrics.get("timeout") is True:
                logger.warning("Evaluation timeout detected", 
                             program_id=program_id,
                             timeout_duration=self.config.evaluator.timeout)
                if state.all_programs.get_program(program_id).artifacts_json is not None:
                    artifact_update = {}

                artifact_update={ # 注意这里是要更新的 而不是新建
                        "timeout": True,
                        "timeout_duration": self.config.evaluator.timeout,
                        "failure_stage": "evaluation",
                        "error_type": "timeout",
                    }

            # Add LLM feedback if configured
            llm_eval_result = None
            if self.config.evaluator.use_llm_feedback and self.llm_evalutor:
                logger.step("Adding LLM feedback to evaluation", program_id=program_id)
                llm_result = await self._llm_evaluate(program_code) # 返回一个dict  {'readability': 0.85, 'maintainability': 0.8, 'efficiency': 0.7, 'other_information': ''}
                llm_eval_result = self._process_evaluation_result(llm_result)

                for name, value in llm_result.items():
                    eval_result.metrics[f"llm_{name}"] = value
                
                logger.info("LLM feedback added to evaluation", 
                           program_id=program_id,
                           llm_metrics_count=len([k for k in eval_result.metrics.keys() if k.startswith('llm_')]))

            # Store artifacts if enabled and present
            if (
                artifacts_enabled
                and (
                    eval_result.has_artifacts()
                    or (llm_eval_result and llm_eval_result.has_artifacts())
                )
                and program_id
            ):
                logger.step("Processing evaluation artifacts", program_id=program_id)
                if state.all_programs.get_program(program_id).artifacts_json is not None:
                    artifact_update = {}

                # Merge eval_result artifacts with llm artifacts if they exist
                if eval_result.has_artifacts():
                    artifact_update=eval_result.artifacts
                    logger.debug("Evaluation artifacts merged", 
                               program_id=program_id,
                               artifacts_count=len(eval_result.artifacts))

                if llm_eval_result and llm_eval_result.has_artifacts():
                    artifact_update=llm_eval_result.artifacts
                    logger.debug("LLM artifacts merged", 
                               program_id=program_id,
                               artifacts_count=len(llm_eval_result.artifacts))

            elapsed = time.time() - start_time
            logger.info("Program evaluation completed successfully", 
                       program_id=program_id,
                       elapsed_time=f"{elapsed:.2f}s",
                       final_metrics_count=len(eval_result.metrics))
            
            # Return just metrics for backward compatibility
            return eval_result , artifact_update

        except asyncio.TimeoutError: #如果是超时错误 
            # Handle timeout specially - don't retry, just return timeout result
            logger.error("Evaluation timeout occurred", 
                        program_id=program_id,
                        timeout_duration=self.config.evaluator.timeout)

            # Capture timeout artifacts if enabled
            if artifacts_enabled and program_id:
                artifact_update = {
                    "timeout": True,
                    "timeout_duration": self.config.evaluator.timeout,
                    "failure_stage": "evaluation",
                    "error_type": "timeout",
                }
                eval_result.metrics.update({"error": 0.0,"timeout": True})

            return eval_result , artifact_update

        except Exception as e:
            last_exception = e
            logger.error("Evaluation failed with exception", 
                        program_id=program_id,
                        error=str(e),
                        error_type=type(e).__name__)

            # Capture failure artifacts if enabled
            if artifacts_enabled and program_id:
                artifact_update = {
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    "failure_stage": "evaluation",
                }

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.debug("Temporary file cleaned up", temp_file_path=temp_file_path)

        # All retries failed
        logger.error("All evaluation attempts failed", 
                    program_id=program_id,
                    last_error=str(last_exception) if last_exception else "Unknown")
        eval_result.metrics.update({"error": 0.0})
        return eval_result , artifact_update

class node_sample_parent_inspiration(SyncNode):
    '''
    采样父代程序与灵感程序节点
    
    功能描述：
    - 使用多种策略采样父代程序：探索性采样、利用性采样、随机采样
    - 根据精英比例、多样性和性能采样灵感程序
    - 支持MAP-Elites特征网格的多样性维护
    - 平衡探索与利用，避免过早收敛
    
    采样策略：
    1. 探索性采样：从当前岛屿随机选择，维持多样性
    2. 利用性采样：从精英归档选择，利用优秀基因
    3. 随机采样：完全随机选择，增加不确定性
    
    灵感程序来源：
    1. 全局最优程序（如果与父代不同）
    2. 顶级程序（按精英选择比例）
    3. 多样性程序（从特征网格附近采样）
    4. 随机程序（填充剩余位置）
    
    更新的状态字段：
    - sample_program: 选中的父代程序对象
    - sample_inspirations: 灵感程序ID列表
    - status: 节点状态更新为SAMPLE
    
    输入要求：
    - IslandState: 包含程序库和归档的岛屿状态
    
    输出结果：
    - Dict: 包含采样结果的字典
    '''
    def __init__(self,config:Config,n:int=5,island_id:str = '0',metric:Optional[str] = None):
        self.config = config
        self.n = n
        self.metric = metric
        self.island_id = f"Island_{island_id}" #成员变量的名称
        self.client = None
    def execute(self,state:IslandState)->Tuple[Program,List[Program]]:
        logger.step("Starting parent and inspiration sampling", island_id=state.id)
        
        try:
            logger.debug("Sampling parent program", island_id=state.id)
            parent_program = self._sample_parent(state)
            if parent_program is None:
                logger.error("Parent program sampling failed, returned None", island_id=state.id)
                return None,[]
            
            parent_id = parent_program.id
            logger.info("Parent program sampled successfully", 
                       island_id=state.id,
                       parent_id=parent_id)
            
        except Exception as e:
            logger.error("Parent program sampling failed with exception", 
                        island_id=state.id,
                        error=str(e),
                        error_type=type(e).__name__)
            return None,[]

        logger.debug("Sampling inspiration programs", 
                    island_id=state.id,
                    parent_id=parent_id,
                    target_count=self.n)
        inspirations = self._sample_inspirations(state,parent_id)
        logger.info("Inspiration programs sampled successfully", 
                   island_id=state.id,
                   inspiration_count=len(inspirations),
                   inspiration_ids=inspirations)
        return parent_program,inspirations

    def __call__(self, state: IslandState, config: Optional[Any] = None) -> Dict[str, Any]:
        logger.step("node_sample_parent_inspiration __call__ method invoked", island_id=state.id)
        
        try:
            # 执行采样逻辑
            parent_program,inspirations = self.execute(state)
            logger.info("node_sample_parent_inspiration __call__ method completed successfully", 
                       island_id=state.id,
                       parent_sampled=parent_program is not None,
                       inspiration_count=len(inspirations) if inspirations else 0)
            
            # 只返回需要更新的字段，让LangGraph的reducer处理并发
            if self.client is None:
                self.client = SimpleClient(self.config.port)
            
            update_dict = {
                "sample_program":parent_program,
                "sample_inspirations":inspirations,
                "status":IslandStatus.BUILD_PROMPT
            }
            
            self.client.send_message({
                    "island_id":state.id,
                    "update_dict":update_dict,
                    
                })
            logger.info("node_sample_parent_inspiration __call__ method completed successfully", 
                       island_id=state.id,)
            return update_dict
        except Exception as e:
            logger.error("node_sample_parent_inspiration __call__ method failed", 
                        island_id=state.id,
                        error=str(e))
            raise

    def _sample_inspirations(self,state:IslandState,parent_id:str):
        '''
        采样灵感程序

        采样灵感程序用于下一轮进化

        灵感程序用于指导进化过程，包括：
        1. 绝对最优程序（如果与父代不同）
        2. 顶级程序（按精英选择比例）
        3. 多样性程序（从附近特征格子采样）
        4. 随机程序（填充剩余位置）

        Args:
            parent: 父代程序
            n: 灵感程序数量

        Returns:
            List[Program]: 灵感程序列表

        '''
        logger.step("Starting inspiration sampling", 
                   island_id=state.id,
                   parent_id=parent_id,
                   target_count=self.n)
        
        inspirations = []

        #若最优程序存在 且与父代不同 且在所有的程序中 则加入灵感程序
        if (
            state.all_best_program is not None
            and state.all_best_program.id != parent_id
            and state.all_best_program.id in state.all_programs.get_program_ids()
        ):
            inspirations.append(state.all_best_program.id)
            logger.debug("Added best program to inspirations", 
                        island_id=state.id,
                        best_program_id=state.all_best_program.id)

        # 添加顶级程序作为灵感
        top_n = max(1, int(self.n * self.config.island.elite_selection_ratio))
        logger.debug("Sampling top programs", 
                    island_id=state.id,
                    top_n=top_n,
                    elite_ratio=self.config.island.elite_selection_ratio)
        
        top_programs = get_top_programs(state,n=top_n,metric=self.metric)
        for program in top_programs:
            if program.id not in inspirations and program.id != parent_id:
                inspirations.append(program.id)
                logger.debug("Added top program to inspirations", 
                           island_id=state.id,
                           top_program_id=program.id)

        # 添加多样性程序
        if len(state.all_programs) > self.n and len(inspirations) < self.n:
            logger.step("Sampling diversity programs", 
                       island_id=state.id,
                       remaining_slots=self.n - len(inspirations))
            
            # 计算要添加的多样性程序数量（最多到剩余位置）
            remaining_slots = self.n - len(inspirations)

            # 从不同的特征格子采样以获得多样性
            feature_coords = _calculate_feature_coords(self.config,state,state.all_programs.get_program(parent_id))#这里获得的是一个描述父代多样性的坐标
            logger.debug("Calculated feature coordinates", 
                        island_id=state.id,
                        feature_coords=feature_coords)
            
            # 从父代附近的特征格子获取程序
            nearby_programs = []
            for _ in range(remaining_slots):
                # 扰动坐标
                perturbed_coords = [
                    max(0, min(self.config.island.feature_bins - 1, c + random.randint(-1, 1)))
                    for c in feature_coords
                ]

                # 尝试从这个格子获取程序
                cell_key = _feature_coords_to_key(perturbed_coords) #这里获得的是一个多样性的坐标
                if cell_key in state.feature_map: #如果这个坐标在特征网格中 则获取这个与父代多样性相似的程序
                    program_id = state.feature_map[cell_key]
                    # 在添加前检查程序是否仍然存在
                    if (
                        program_id != parent_id
                        and program_id not in inspirations
                        and program_id in state.all_programs
                    ):
                        nearby_programs.append(state.all_programs.get_program(program_id).id)
                        logger.debug("Added diversity program to inspirations", 
                                   island_id=state.id,
                                   diversity_program_id=program_id)
            
            # 如果需要更多，添加随机程序
            if len(inspirations) + len(nearby_programs) < self.n:
                remaining = self.n - len(inspirations) - len(nearby_programs)
                all_ids = set(state.all_programs.get_program_ids())
                excluded_ids = (
                    {parent_id}
                    .union(inspirations)
                    .union(nearby_programs)
                )
                available_ids = list(all_ids - excluded_ids)

                if available_ids:
                    random_ids = random.sample(available_ids, min(remaining, len(available_ids)))
                    inspirations.extend(random_ids)
                    logger.debug("Added random programs to inspirations", 
                               island_id=state.id,
                               random_count=len(random_ids))
            
            inspirations.extend(nearby_programs)
            
        # 保持顺序去重 inspirations
        seen = set()
        inspirations = [x for x in inspirations if not (x in seen or seen.add(x))]
        logger.info("Inspiration sampling completed", 
                   island_id=state.id,
                   final_count=len(inspirations[:self.n]),
                   inspiration_ids=inspirations[:self.n])
        
        return inspirations[:self.n]
    def _sample_parent(self,state:IslandState) -> Optional[Program | None | Any]:
        """
        从当前岛屿采样父代程序用于下一轮进化

        使用多种采样策略：
        1. 探索（exploration）：从当前岛屿采样
        2. 利用（exploitation）：从精英归档采样
        3. 随机（random）：从所有程序中随机采样

        Returns:
            Program: 选中的父代程序
        """
        logger.step("Starting parent sampling strategy selection", island_id=state.id)
        
        # 使用探索比例和利用比例决定采样策略
        rand_val = random.random()
        logger.debug("Sampling strategy random value", 
                    island_id=state.id,
                    rand_val=rand_val,
                    exploration_ratio=self.config.island.exploration_ratio,
                    exploitation_ratio=self.config.island.exploitation_ratio)

        if rand_val < self.config.island.exploration_ratio:
            logger.info("Using exploration sampling strategy", island_id=state.id)
            # 探索：从当前岛屿采样（多样性采样）
            return self._sample_exploration_parent(state)
        elif rand_val < self.config.island.exploration_ratio + self.config.island.exploitation_ratio:
            logger.info("Using exploitation sampling strategy", island_id=state.id)
            # 利用：从归档采样（精英程序）
            return self._sample_exploitation_parent(state)
        else:
            logger.info("Using random sampling strategy", island_id=state.id)
            # 随机：从任何程序采样（剩余概率）
            return self._sample_random_parent(state)
        
    def _sample_exploration_parent(self, state: IslandState) -> Optional[Program | None | Any]:
        """
        探索性采样父代程序（从当前岛屿采样策略）

        该方法实现了岛屿模型中的探索性采样策略，用于从当前岛屿的程序池中
        随机选择一个程序作为下一轮进化的父代程序。这种采样方式有助于维持
        种群的多样性，避免过早收敛到局部最优解。

        算法流程：
        1. 获取当前岛屿的程序ID列表
        2. 验证岛屿是否正确初始化（非空检查）
        3. 从程序列表中随机选择一个程序ID
        4. 根据ID从全局程序字典中获取对应的Program对象

        并行模式说明：
        在并行执行模式下，每个子图负责处理一个特定的岛屿，state.current_island
        字段指示当前子图所处理的岛屿索引。这种设计确保了各岛屿间的独立性
        和并行处理的正确性。

        Args:
            state (GraphState): 图状态对象，包含所有岛屿的程序信息和当前岛屿索引

        Returns:
            Program: 从当前岛屿随机选择的父代程序对象

        Raises:
            ValueError: 当前岛屿程序列表为空，表示岛屿未正确初始化
            KeyError: 程序ID在全局程序字典中不存在（理论上不应发生）

        Note:
            - 每个岛屿在初始化时都应包含至少一个初始程序
            - 该方法是探索-利用权衡策略的一部分
            - 随机采样有助于维持遗传算法的多样性
        """
        logger.step("Starting exploration parent sampling", island_id=state.id)
        
        # 获取当前岛屿的程序ID列表
        # current_island索引对应当前子图处理的岛屿
        current_island_programs = state.programs.get_program_ids()
        logger.debug("Retrieved current island programs", 
                    island_id=state.id,
                    program_count=len(current_island_programs))

        # 岛屿完整性验证
        # 正常情况下每个岛屿都应该包含至少一个初始程序
        if not current_island_programs:
            logger.error("Island not properly initialized, program list is empty", 
                        island_id=state.id)
            raise ValueError(
                f"岛屿 {state.id} 未正确初始化，程序列表为空。"
                f"请确保初始程序已正确设置到所有岛屿中。"
            )

        # 探索性随机采样
        # 使用random.choice确保每个程序被选中的概率相等
        parent_id = random.choice(list(current_island_programs))
        logger.info("Exploration parent selected", 
                   island_id=state.id,
                   selected_parent_id=parent_id)

        # 从全局程序字典中获取完整的Program对象
        # 这里假设程序ID的一致性已经在其他地方得到保证
        selected_program = state.all_programs.get_program(parent_id)
        logger.debug("Exploration parent program retrieved", 
                    island_id=state.id,
                    parent_id=parent_id)

        return selected_program

    def _sample_exploitation_parent(self, state: IslandState) -> Optional[Program | None | Any]:
        """
        利用性采样父代程序（从精英归档采样策略）

        该方法实现了岛屿模型中的利用性采样策略，用于从精英归档中
        选择一个程序作为下一轮进化的父代程序。这种采样方式有助于
        利用已有的优秀程序，加速收敛到全局最优解。

        算法流程：
        """
        logger.step("Starting exploitation parent sampling", island_id=state.id)

        archive_programs_ids = state.archive.get_program_ids() # 精英归档的id集合 archive在初始化时候会被初始化 
        logger.debug("Retrieved archive programs", 
                    island_id=state.id,
                    archive_count=len(archive_programs_ids))
        
        # 尽可能获得当前岛屿内部的精英归档
        archive_in_current_island = [pid for pid in archive_programs_ids if pid in state.programs.get_program_ids()]
        logger.debug("Filtered archive programs in current island", 
                    island_id=state.id,
                    current_island_archive_count=len(archive_in_current_island))
        
        #优先从自己的岛屿上采样 如果自己的岛屿上没有精英归档类别的程序 就随机采样
        if len(archive_in_current_island) > 0: #如果自己的岛屿上有精英归档类别的程序 则从自己的岛屿上采样
            selected_parent_id = random.choice(archive_in_current_island)
            logger.info("Selected exploitation parent from current island archive", 
                       island_id=state.id,
                       selected_parent_id=selected_parent_id)
            return state.all_programs.get_program(selected_parent_id)
        else: #如果自己的岛屿上没有精英归档类别的程序 则从所有岛屿的精英归档中采样
            if len(archive_programs_ids) > 0:
                selected_parent_id = random.choice(archive_programs_ids)
                logger.info("Selected exploitation parent from global archive", 
                           island_id=state.id,
                           selected_parent_id=selected_parent_id)
                return state.all_programs.get_program(selected_parent_id)
            else:
                logger.warning("No archive programs available, falling back to random sampling", 
                             island_id=state.id)
                return self._sample_random_parent(state)

    def _sample_random_parent(self,state:IslandState) -> Optional[Program | None | Any]:
        """
        完全随机采样父代程序

        Returns:
            Program: 选中的父代程序
        """
        logger.step("Starting random parent sampling", island_id=state.id)
        
        if not state.all_programs:
            logger.error("No programs available for random sampling", island_id=state.id)
            raise ValueError("No programs available for sampling")

        # 从自己岛屿上的所有程序中随机采样
        current_island_programs = state.programs.get_program_ids()
        logger.debug("Retrieved current island programs for random sampling", 
                    island_id=state.id,
                    program_count=len(current_island_programs))
        
        if not current_island_programs:
            logger.error("Current island program list is empty", island_id=state.id)
            raise ValueError(f"当前岛屿 {state.id} 的程序列表为空")
        
        program_id = random.choice(list(current_island_programs))
        logger.info("Random parent selected", 
                   island_id=state.id,
                   selected_parent_id=program_id)
        
        selected_program = state.all_programs.get_program(program_id)
        logger.debug("Random parent program retrieved", 
                    island_id=state.id,
                    parent_id=program_id)
        
        return selected_program

class node_build_prompt(SyncNode):
    '''
    构建LLM提示词节点
    
    功能描述：
    - 根据岛屿状态信息构建用于LLM生成的提示词
    - 整合当前程序、父代程序、历史程序信息
    - 包含顶级程序和灵感程序作为进化参考
    - 支持基于差异和基于重写两种进化模式
    - 构建包含程序工件、指标、语言等上下文信息的完整提示
    
    提示词组成部分：
    - current_program: 当前程序代码
    - parent_program: 父代程序代码和指标
    - previous_programs: 前三代程序历史
    - top_programs: 按性能排序的顶级程序（前5个）
    - inspirations: 灵感程序列表
    - program_artifacts: 程序相关工件信息
    - evolution_round: 当前演化轮次
    - language: 编程语言类型
    
    更新的状态字段：
    - prompt: 构建好的LLM提示词字符串
    - status: 节点状态更新为BUILD_PROMPT
    
    输入要求：
    - IslandState: 包含程序信息和采样结果的岛屿状态
    
    输出结果：
    - Dict: 包含提示词和状态的字典
    
    注意事项：
    - 初始化时inspiration可能为空列表
    - 会处理程序不存在的异常情况
    '''
    def __init__(self,config:Config,metric:Optional[str] = None):
        self.config = config
        self.metric = metric
        self.prompt_sampler = PromptSampler_langchain(config=config.prompt)
        self.client = None
    def execute(self,state:IslandState):
        logger.step("Starting prompt building", island_id=state.id)
        prompt = self._build_prompt(state)
        logger.info("Prompt building completed", 
                   island_id=state.id,
                   prompt_length=len(prompt) if prompt else 0)
        return prompt
    def __call__(self, state: IslandState, config: Optional[Any] = None) -> Dict[str, Any]:
        logger.step("node_build_prompt __call__ method invoked", island_id=state.id)
        
        try:
            prompt = self.execute(state)
            logger.info("node_build_prompt __call__ method completed successfully", 
                       island_id=state.id,
                       prompt_created=bool(prompt))
            
            if self.client is None:
                self.client = SimpleClient(self.config.port)
            
            update_dict = {
                "prompt": prompt,
                "status": IslandStatus.LLM_GENERATE
            }
            
            self.client.send_message({
                    "island_id":state.id,
                    "update_dict":update_dict,
                    
                })
            logger.info("node_build_prompt __call__ method completed successfully", 
                       island_id=state.id,
                       prompt_created=bool(prompt))
            return update_dict
        except Exception as e:
            logger.error("node_build_prompt __call__ method failed", 
                        island_id=state.id,
                        error=str(e))
            raise
    def _build_prompt(self,state:IslandState)->str:
        logger.step("Starting prompt construction", island_id=state.id)

        # 若当前program为空 则使用父代程序
        current_program_id = state.latest_program.id if state.latest_program.id is not None else state.sample_program.id
        current_program = state.all_programs.get(current_program_id)
        
        logger.debug("Retrieved current program", 
                    island_id=state.id,
                    current_program_id=current_program_id,
                    current_program_exists=current_program is not None)
        
        if current_program is None:
            logger.error("Current program is empty", 
                        island_id=state.id,
                        current_program_id=current_program_id)
            return ""

        parent_program = state.all_programs.get(state.sample_program.id)
        logger.debug("Retrieved parent program", 
                    island_id=state.id,
                    parent_program_id=state.sample_program.id,
                    parent_program_exists=parent_program is not None)
        
        if parent_program is None:
            logger.error("Parent program is empty", 
                        island_id=state.id,
                        parent_program_id=state.sample_program.id)
            return ""
        
        parent_code = parent_program.code
        parent_metrics = parent_program.metrics
        logger.debug("Extracted parent program information", 
                    island_id=state.id,
                    parent_code_length=len(parent_code),
                    parent_metrics_count=len(parent_metrics))

        logger.step("Building previous programs history", island_id=state.id)
        previous_programs = []
        #当前的父代程序

        for i in range(3):
            #如果父代程序有父代程序 则将父代程序的父代程序加入previous_programs 否则结束循环
            if isinstance(parent_program,Program) and parent_program.parent_id:
                previous_program = state.all_programs.get(parent_program.parent_id)
                if previous_program:
                    previous_programs.append(previous_program)
                    logger.debug("Added previous program to history", 
                               island_id=state.id,
                               generation=i+1,
                               previous_program_id=parent_program.parent_id)
                parent_program = previous_program
            else:
                logger.debug("Reached end of program history", 
                           island_id=state.id,
                           history_depth=i)
                break

        logger.step("Processing inspiration programs", island_id=state.id)
        inspirations = state.sample_inspirations
        logger.debug("Retrieved inspiration programs", 
                    island_id=state.id,
                    inspiration_count=len(inspirations))
        
        inspirations_programs = [state.all_programs.get_program(ins).to_dict() for ins in inspirations if ins is not None]
        logger.debug("Converted inspiration programs to dict format", 
                    island_id=state.id,
                    converted_count=len(inspirations_programs))

        logger.step("Retrieving top programs", island_id=state.id)
        top_programs = [i.to_dict() for i in get_top_programs(state,n=5,metric=self.metric)]
        logger.debug("Retrieved top programs", 
                    island_id=state.id,
                    top_program_count=len(top_programs))
        
        logger.step("Building final prompt", island_id=state.id)
        prompt = self.prompt_sampler.build_prompt(
            current_program = current_program.code,
            parent_program = parent_code,
            program_metrics = parent_metrics,  # 程序指标字典
            previous_protgrams = previous_programs,  # 之前的程序尝试列表 即这个岛屿上的父代程序的集合 取三代 （即当前父代的前二代）
            top_programs = top_programs,       # 顶级程序列表（按性能排序）
            inspirations = inspirations_programs,       # 灵感程序列表
            language = "python",            # 编程语言
            evolution_round = state.iteration,            # 演化轮次
            diff_based_evolution = self.config.diff_based_evolution,   # 是否使用基于差异的演化
            program_artifacts = get_artifacts(state,current_program.id),  # 程序工件
        )
        
        logger.info("Prompt construction completed", 
                   island_id=state.id,
                   prompt_length=len(prompt),
                   evolution_round=state.iteration,
                   diff_based_evolution=self.config.diff_based_evolution)
        
        return prompt

class node_llm_generate(SyncNode):
    '''
    LLM程序生成节点
    
    功能描述：
    - 使用LLM根据构建的提示词生成子代程序
    - 支持两种进化模式：基于差异(diff)和基于重写(rewrite)
    - 处理LLM生成失败、无差异等异常情况
    - 创建包含新代码的Program对象
    - 记录变更摘要和生成建议
    
    生成模式：
    1. 基于差异模式：
       - 生成代码差异(diff_code)和建议(suggestion)
       - 使用apply_diff将差异应用到父代程序
       - 验证差异块的有效性
    
    2. 基于重写模式：
       - 生成完整的重写代码(rewrite_code)
       - 解析并验证新代码的完整性
       - 标记变更摘要为"full rewrite"
    
    错误处理：
    - LLM响应为None时触发重新采样
    - 差异块为空时触发重新采样
    - 重写代码无效时触发重新采样
    - 失败时增加迭代计数和会议进度
    
    更新的状态字段（成功时）：
    - diff_message/rewrite_message: LLM生成的代码内容
    - suggestion_message: LLM提供的改进建议
    - status: 节点状态更新为LLM_GENERATE
    - llm_generate_success: 生成成功标志（True）
    - latest_program: 包含新代码的Program对象
    - change_summary: 变更摘要
    
    更新的状态字段（失败时）：
    - status: 保持LLM_GENERATE状态
    - llm_generate_success: 生成失败标志（False）
    - iteration: 迭代次数+1
    - now_meeting: 会议进度+1
    - next_meeting: 下次会议时间-1
    
    输入要求：
    - IslandState: 包含prompt的岛屿状态
    
    输出结果：
    - Dict: 包含生成结果和状态的字典
    
    失败重试机制：
    - 生成失败时会触发重新采样父代程序
    - 重新构建提示词并再次尝试生成
    '''
    def __init__(self,config:Config):
        self.config = config
        # self.island_id = island_id
        self.llm = LLMs(config=self.config)
        self.diff_based_evolution = self.config.diff_based_evolution
        if self.diff_based_evolution: #如果是基于差异的演化 那么使用
            self.structure = ResponseFormatter_template_diff
            self.key = ["suggestion","diff_code"]
            logger.info("LLM generation configured for diff-based evolution")
        else:
            self.structure = ResponseFormatter_template_rewrite
            self.key = ["suggestion","rewrite_code"]
            logger.info("LLM generation configured for rewrite-based evolution")
        self.client =  None

    async def _llm_generate(self,state:IslandState):
        return await self.llm.ainvoke(state.prompt,self.structure,self.key)
    async def execute(self,state:IslandState):
        logger.step("Starting LLM program generation", 
                   island_id=state.id,
                   evolution_type="diff-based" if self.diff_based_evolution else "rewrite-based")
        
        llm_response:Optional[Dict[str,Any]|None] = await self._llm_generate(state)
        logger.debug("LLM response received", 
                    island_id=state.id,
                    response_type=type(llm_response),
                    response_keys=list(llm_response.keys()) if llm_response else [])
        
        generation = state.iteration
        if self.client is None:
            self.client = SimpleClient(self.config.port)
        
        if llm_response is None:
            # 失败 
            logger.error("LLM generation failed, response is None", island_id=state.id)
            update_dict = {
                "status": IslandStatus.SAMPLE,
                "llm_generate_success":False,
                "iteration":generation+1,#重新采样 代数+1
                "now_meeting":state.now_meeting+1,
                "next_meeting":state.next_meeting-1,
            }
            self.client.send_message({
                "island_id":state.id,
                "update_dict":update_dict,
                
            })
            return update_dict
        if self.diff_based_evolution:
            logger.step("Processing diff-based evolution", island_id=state.id)
            parent_program = state.all_programs.get_program(state.sample_program.id)
            if parent_program is None:
                logger.error("Parent program is empty for diff-based evolution", 
                           island_id=state.id,
                           parent_program_id=state.sample_program.id)
                update_dict = {
                    "llm_generate_success":False,
                    "iteration":generation+1,#重新采样 代数+1
                    "now_meeting":state.now_meeting+1,
                    "next_meeting":state.next_meeting-1,
                }
                self.client.send_message({
                    "island_id":state.id,
                    "update_dict":update_dict,
                    
                })
                return update_dict
            
            parent_code = parent_program.code 
            diff_code = llm_response["diff_code"]
            suggestion = llm_response["suggestion"]
            logger.debug("Extracted diff code and suggestion", 
                        island_id=state.id,
                        diff_code_length=len(diff_code),
                        suggestion_length=len(suggestion))
            
            diff_blocks = extract_diffs(diff_code)
            logger.debug("Extracted diff blocks", 
                        island_id=state.id,
                        diff_blocks_count=len(diff_blocks) if diff_blocks else 0)
            
            if not diff_blocks or diff_blocks == []:
                logger.warning("No diff blocks found in LLM response", 
                             island_id=state.id,
                             generation=generation)
                update_dict = {
                    "llm_generate_success":False,
                    "iteration":generation+1,#重新采样 代数+1
                    "now_meeting":state.now_meeting+1,
                    "next_meeting":state.next_meeting-1,
                }
                self.client.send_message({
                    "island_id":state.id,
                    "update_dict":update_dict,
                    
                })
                return update_dict
            
            logger.step("Applying diff to parent code", island_id=state.id)
            child_code = apply_diff(parent_code,diff_code)
            change_summary = format_diff_summary(diff_blocks)
            logger.debug("Diff applied successfully", 
                        island_id=state.id,
                        child_code_length=len(child_code),
                        change_summary=change_summary)
            
            #生成子代的id
            child_id = str(uuid.uuid4())
            logger.info("Child program created successfully", 
                       island_id=state.id,
                       child_id=child_id,
                       parent_id=parent_program.id)
            
            child_Program = Program(id=child_id,
                                    code=child_code,
                                    parent_id=parent_program.id,
                                    )

            update_dict = {
                "diff_message": diff_code,
                "suggestion_message": suggestion,
                "status": IslandStatus.EVALUATE_CHILD,
                "llm_generate_success":True,
                "latest_program":child_Program,
                "change_summary":change_summary,
            }
            self.client.send_message({
                "island_id":state.id,
                "update_dict":update_dict,
                
            })
            return update_dict
        else :  # 基于重写的演化
            logger.step("Processing rewrite-based evolution", island_id=state.id)
            rewrite_code = llm_response["rewrite_code"]
            logger.debug("Extracted rewrite code", 
                        island_id=state.id,
                        rewrite_code_length=len(rewrite_code) if rewrite_code else 0)
            
            if rewrite_code is None or rewrite_code == "":
                logger.warning("No rewrite code found in LLM response", 
                             island_id=state.id,
                             generation=generation)
                update_dict = {
                    "status": IslandStatus.SAMPLE,
                    "llm_generate_success":False,
                    "iteration":generation+1,#重新采样 代数+1
                    "now_meeting":state.now_meeting+1,
                    "next_meeting":state.next_meeting-1,
                }
                self.client.send_message({
                    "island_id":state.id,
                    "update_dict":update_dict,
                    
                })
                return update_dict
            
            logger.step("Parsing full rewrite code", island_id=state.id)
            new_code = parse_full_rewrite(rewrite_code,state.language)
            logger.debug("Parsed rewrite code", 
                        island_id=state.id,
                        new_code_length=len(new_code) if new_code else 0)
            
            if not new_code:
                logger.warning("Failed to parse rewrite code", 
                             island_id=state.id,
                             generation=generation)
                return {
                        "status": IslandStatus.SAMPLE,
                        "llm_generate_success":False,
                        "iteration":generation+1,#重新采样 代数+1
                        "now_meeting":state.now_meeting+1,
                        "next_meeting":state.next_meeting-1,
                    }
            
            change_summary = "full rewrite"
            child_id = str(uuid.uuid4())
            logger.info("Child program created successfully from rewrite", 
                       island_id=state.id,
                       child_id=child_id,
                       parent_id=state.sample_program.id)
            
            child_program = Program(id=child_id,code=new_code,parent_id=state.sample_program.id)
            update_dict = {
                "rewrite_message":rewrite_code,
                "suggestion_message":llm_response["suggestion"],
                "status": IslandStatus.EVALUATE_CHILD,
                "llm_generate_success":True,
                "latest_program":child_program,
                "change_summary":change_summary,
            }
            self.client.send_message({
                "island_id":state.id,
                "update_dict":update_dict,
                
            })
            return update_dict
    def __call__(self,state:IslandState):
        logger.step("node_llm_generate __call__ method invoked", island_id=state.id)
        
        try:
            result = asyncio.run(self.execute(state))
            logger.info("node_llm_generate __call__ method completed successfully", 
                       island_id=state.id,
                       generation_success=result.get("llm_generate_success", False))
            return result
        except Exception as e:
            logger.error("node_llm_generate __call__ method failed", 
                        island_id=state.id,
                        error=str(e))
            raise
  
class node_update(SyncNode):
    '''
    程序库更新节点
    
    功能描述：
    - 将评估完成的新程序添加到各种程序库中
    - 更新全局最佳程序和岛屿最佳程序
    - 维护精英程序归档（固定大小）
    - 更新MAP-Elites特征网格映射
    - 管理岛屿程序库容量（到达上限时替换最差程序）
    - 更新迭代计数和会议进度
    
    更新策略：
    1. 最佳程序更新：
       - 比较当前程序与全局最佳程序
       - 比较当前程序与岛屿最佳程序
       - 使用_is_better函数进行性能比较
    
    2. 程序库管理：
       - 岛屿程序库未满时直接添加
       - 已满时与最差程序比较，优者替换劣者
       - 同步更新全局程序库
    
    3. 精英归档维护：
       - 归档未满时直接添加
       - 已满时与最差归档程序比较并替换
    
    4. 特征网格更新：
       - 计算程序的特征坐标
       - 更新MAP-Elites网格映射
       - 支持多样性维护
    
    更新的状态字段：
    - all_best_program: 全局最佳程序（如果有更新）
    - best_program: 岛屿最佳程序（如果有更新）
    - programs: 岛屿程序库更新操作（"add"或"replace"）
    - latest_program: 当前最新程序
    - archive: 精英归档更新操作（如果有更新）
    - feature_map: 特征网格更新（如果有更新）
    - all_programs: 全局程序库更新操作
    - iteration: 迭代次数+1
    - now_meeting: 会议进度+1
    - next_meeting: 下次会议时间-1
    - status: 节点状态更新为UPDATE
    
    输入要求：
    - IslandState: 包含latest_program的岛屿状态
    
    输出结果：
    - Dict: 包含所有更新操作的字典
    
    性能优化：
    - 只更新需要变更的字段
    - 使用元组操作标记更新类型
    - 避免不必要的数据复制
    '''
    def __init__(self,config:Config):
        self.config = config
        self.client = None
    def __call__(self,state:IslandState):
        logger.step("node_update __call__ method invoked", island_id=state.id)
        
        try:
            update_dict = self.execute(state)
            logger.info("node_update __call__ method completed successfully", 
                       island_id=state.id,
                       update_fields_count=len(update_dict))
            
            if self.client is None:
                self.client = SimpleClient(self.config.port)
            
            self.client.send_message({
                    "island_id":state.id,
                    "update_dict":update_dict,
                    
                })
            return update_dict
        except Exception as e:
            logger.error("node_update __call__ method failed", 
                        island_id=state.id,
                        error=str(e))
            raise
    
    def execute(self,state:IslandState):
        logger.step("Starting program library update", 
                   island_id=state.id,
                   current_program_id=state.latest_program.id)
        
        # 此时all_programs island_program 中还没有新的program 
        current_program = state.latest_program
        logger.debug("Retrieved current program for update", 
                    island_id=state.id,
                    current_program_id=current_program.id)
        
        # 与所有程序的副本相比 更好的程序
        logger.step("Checking global best program update", island_id=state.id)
        best_program = self._best_program_update(state) 
        logger.debug("Global best program check completed", 
                    island_id=state.id,
                    best_program_updated=best_program is not None)
        
        # 与当前岛屿上的程序相比 更好的程序
        logger.step("Checking island best program update", island_id=state.id)
        best_program_each_island = self._best_program_this_island(state)
        logger.debug("Island best program check completed", 
                    island_id=state.id,
                    island_best_updated=best_program_each_island is not None)
        
        logger.step("Updating island programs", island_id=state.id)
        island_programs_update = self._update_island_programs(state)
        logger.debug("Island programs update completed", 
                    island_id=state.id,
                    update_type=island_programs_update[0] if island_programs_update else None)
        
        # 是否更新all_programs 取决于island_programs 
        all_programs_update = None 
        if island_programs_update is None:
            all_programs_update = None 
        elif isinstance(island_programs_update,tuple):
            all_programs_update = island_programs_update
        else:
            logger.error("Invalid island_programs_update format", 
                        island_id=state.id,
                        update_type=type(island_programs_update),
                        update_length=len(island_programs_update))
            raise ValueError("island_programs_update must be a tuple of length 2 or 4, but you give me a {}".format(len(island_programs_update)))

        logger.step("Updating newest program", island_id=state.id)
        newest_program_update = self._update_newest_program(state)
        
        logger.step("Updating archive", island_id=state.id)
        archive_update = self._update_archive(current_program,state)
        logger.debug("Archive update completed", 
                    island_id=state.id,
                    archive_updated=archive_update is not None)
        
        logger.step("Updating feature map", island_id=state.id)
        feature_map_update = self._update_feature_map(current_program,state)
        logger.debug("Feature map update completed", 
                    island_id=state.id,
                    feature_map_updated=feature_map_update is not None)
        
        logger.step("Building final update dictionary", island_id=state.id)
        update_dict = {}
        if best_program is not None : 
            update_dict["all_best_program"] = best_program
            logger.info("Global best program updated", 
                       island_id=state.id,
                       new_best_program_id=best_program.id)
            
        if best_program_each_island is not None:
            update_dict["best_program"] = best_program_each_island
            logger.info("Island best program updated", 
                       island_id=state.id,
                       new_island_best_id=best_program_each_island.id)
        
        if island_programs_update is not None:
            update_dict["programs"] = island_programs_update # e.g ("add",self.island_id,current_program)
            logger.info("Island programs updated", 
                       island_id=state.id,
                       update_operation=island_programs_update[0])
       
        update_dict["latest_program"] = newest_program_update
        if archive_update is not None:
            update_dict["archive"] = archive_update
            logger.info("Archive updated", 
                       island_id=state.id,
                       archive_operation=archive_update[0])
            
        if feature_map_update is not None:
            update_dict["feature_map"] = feature_map_update
            logger.info("Feature map updated", 
                       island_id=state.id,
                       feature_key=feature_map_update[0])
        
        if all_programs_update is not None:
            update_dict["all_programs"] = all_programs_update
            logger.info("Global programs updated", 
                       island_id=state.id,
                       global_update_operation=all_programs_update[0])
        
        current_iteration = state.iteration
        
        update_dict["iteration"] = current_iteration + 1
        update_dict["now_meeting"] = state.now_meeting + 1
        update_dict["next_meeting"] = state.next_meeting - 1
        
        update_dict["status"] = IslandStatus.SAMPLE
        
        logger.info("Program library update completed", 
                   island_id=state.id,
                   total_updates=len(update_dict),
                   new_iteration=current_iteration + 1,
                   now_meeting=state.now_meeting,
                   next_meeting=state.next_meeting,)
        
        return update_dict
        
        
        # for k,v in update_dict.items():
        #     logger.info(f"key:{k},value type:{type(v)}")
        ##import pdb;pdb.set_trace()
        
        return update_dict
    def _best_program_update(self,state:IslandState)->Optional[Program]:
        '''
        如果当前程序更好 则返回需要更新的 best_program 和 best_program_id 和 best_metrics
        '''
        logger.step("Checking global best program update", island_id=state.id)
        
        current_program = state.latest_program
        best_program = state.all_best_program
        
        logger.debug("Comparing current program with global best", 
                    island_id=state.id,
                    current_program_id=current_program.id,
                    current_best_id=best_program.id if best_program else None)
        
        if _is_better(current_program,best_program) or best_program is None:#若当前程序更好 或者best_program为空
            logger.info("Global best program updated", 
                       island_id=state.id,
                       new_best_program_id=current_program.id,
                       previous_best_id=best_program.id if best_program else None)
            best_program = current_program
            return best_program
        else:
            logger.debug("Global best program unchanged", 
                        island_id=state.id,
                        current_program_id=current_program.id,
                        best_program_id=best_program.id if best_program else None)
            return None
    def _best_program_this_island(self,state:IslandState)->Optional[Program|None]:
        '''
        获取当前岛屿上最好的程序 并比较 如果当前程序更好 则更新best_program_each_island中[self.island_id]的id
        '''
        logger.step("Checking island best program update", island_id=state.id)
        
        current_program = state.latest_program
        best_program_this_island = state.best_program

        logger.debug("Comparing current program with island best", 
                    island_id=state.id,
                    current_program_id=current_program.id,
                    island_best_id=best_program_this_island.id if best_program_this_island else None)

        if _is_better(current_program,best_program_this_island) or best_program_this_island is None:
            logger.info("Island best program updated", 
                       island_id=state.id,
                       new_island_best_id=current_program.id,
                       previous_island_best_id=best_program_this_island.id if best_program_this_island else None)
            best_program_this_island = current_program
            return best_program_this_island
        else:
            logger.debug("Island best program unchanged", 
                        island_id=state.id,
                        current_program_id=current_program.id,
                        island_best_id=best_program_this_island.id if best_program_this_island else None)
            return None
        
        
        
    def _update_island_programs(self,state:IslandState)->Optional[tuple[str, str, Program | Any | None]|Tuple[str,Program]|None]:
        logger.step("Updating island programs", island_id=state.id)
        
        #更新当前岛屿上的程序 这时候首先要检查岛屿上的程序个数是否超过max_island_programs_size
        max_island_programs_size = self.config.island.population_size
        current_program = state.latest_program
        
        logger.debug("Checking island program capacity", 
                    island_id=state.id,
                    current_program_count=len(state.programs),
                    max_capacity=max_island_programs_size,
                    current_program_id=current_program.id)
        
        if len(state.programs) < max_island_programs_size:
            logger.info("Adding new program to island (capacity available)", 
                       island_id=state.id,
                       current_program_id=current_program.id)
            return ("add",current_program)
        else:
            # 如果岛屿上的程序个数到达了峰值 需要移除某个程序
            # 当前的程序会和岛屿中最差的程序进行对比 如果当前程序更好 则移除最差的程序 并添加当前程序 否则不进行任何操作 
            logger.step("Island at capacity, evaluating replacement", island_id=state.id)
            
            island_programs = state.programs.get_all_programs() # 获取岛屿上的所有程序 Dict{program_id:Program}
            metrics_dict = {pid:program.metrics for pid,program in island_programs.items()}
            worst_program_id = min(metrics_dict,key=lambda x:safe_numeric_average(metrics_dict[x]))
            worst_program = state.programs.get_program(worst_program_id)
            
            logger.debug("Found worst program in island", 
                        island_id=state.id,
                        worst_program_id=worst_program_id,
                        worst_program_metrics=worst_program.metrics if worst_program else None)
            
            if _is_better(current_program,worst_program):
                logger.info("Replacing worst program with current program", 
                           island_id=state.id,
                           current_program_id=current_program.id,
                           replaced_program_id=worst_program_id)
                return ("replace",worst_program_id,current_program)
            else:
                logger.debug("Current program not better than worst, no replacement", 
                            island_id=state.id,
                            current_program_id=current_program.id,
                            worst_program_id=worst_program_id)
                return None
    def _update_newest_program(self,state:IslandState):
        logger.step("Updating newest program", island_id=state.id)
        #更新当前岛屿上的最新程序 
        logger.debug("Newest program updated", 
                    island_id=state.id,
                    newest_program_id=state.latest_program.id)
        return state.latest_program
    def _update_archive(self, program: Program,state:IslandState):
        """
        更新精英程序归档
        
        维护一个固定大小的精英程序集合，用于保存表现最好的程序：
        1. 如果归档未满，直接添加新程序
        2. 如果归档已满，清理无效引用后重新评估
        3. 如果归档仍满，与最差程序比较，优者替换劣者
        
        Args:
            program: 要考虑加入归档的程序对象
        """
        logger.step("Updating archive", island_id=state.id)
        
        logger.debug("Checking archive capacity", 
                    island_id=state.id,
                    current_archive_size=len(state.archive),
                    max_archive_size=self.config.archive_size,
                    candidate_program_id=program.id)
        
        #检查归档是否已满，未满则直接添加
        if len(state.archive) < self.config.archive_size:
            logger.info("Adding program to archive (capacity available)", 
                       island_id=state.id,
                       program_id=program.id)
            return ("add",program)
        
        elif len(state.archive) == self.config.archive_size: #已满 将其与最差的程序对比 并决定是否替换最差的程序
            logger.step("Archive at capacity, evaluating replacement", island_id=state.id)
            
            archive_programs = state.archive.get_all_programs()
            metrics_dict = {pid:program.metrics for pid,program in archive_programs.items()}
            worst_program_id = min(metrics_dict,key=lambda x:safe_numeric_average(metrics_dict[x]))
            worst_program = archive_programs[worst_program_id]
            
            logger.debug("Found worst program in archive", 
                        island_id=state.id,
                        worst_program_id=worst_program_id,
                        worst_program_metrics=worst_program.metrics if worst_program else None)
            
            if _is_better(program,worst_program):
                logger.info("Replacing worst program in archive", 
                           island_id=state.id,
                           new_program_id=program.id,
                           replaced_program_id=worst_program_id)
                return ("replace",worst_program_id,program)
            else:
                logger.debug("Candidate program not better than worst in archive", 
                            island_id=state.id,
                            candidate_program_id=program.id,
                            worst_program_id=worst_program_id)
                return None
    def _update_feature_map(self,program:Program,state:IslandState):
        logger.step("Updating feature map", island_id=state.id)
        
        #更新特征网格
        #计算当前程序在特征网格中的坐标
        feature_coords = _calculate_feature_coords(self.config,state,program)
        logger.debug("Calculated feature coordinates", 
                    island_id=state.id,
                    program_id=program.id,
                    feature_coords=feature_coords)

        # 更新MAP-Elites特征网格
        feature_key = _feature_coords_to_key(feature_coords)
        logger.debug("Generated feature key", 
                    island_id=state.id,
                    feature_key=feature_key)
        
        should_replace = feature_key in state.feature_map #若feature_key在feature_map中，则需要替换
        logger.debug("Checking if feature key exists in map", 
                    island_id=state.id,
                    feature_key=feature_key,
                    key_exists=should_replace)
        
        if should_replace:#若需要替换 取出feature_map中对应的program_id
            program_id_need_replace = state.feature_map[feature_key] #旧的
            logger.debug("Found existing program at feature key", 
                        island_id=state.id,
                        feature_key=feature_key,
                        existing_program_id=program_id_need_replace)
            
            if program_id_need_replace not in state.all_programs: #如果旧的程序id不在all_programs中 则直接替换
                logger.info("Replacing invalid program reference in feature map", 
                           island_id=state.id,
                           feature_key=feature_key,
                           new_program_id=program.id,
                           invalid_program_id=program_id_need_replace)
                return (feature_key,program.id)
            else:
                existing_program = state.all_programs.get_program(program_id_need_replace)
                if _is_better(program,existing_program):
                    logger.info("Replacing program in feature map (better performance)", 
                               island_id=state.id,
                               feature_key=feature_key,
                               new_program_id=program.id,
                               replaced_program_id=program_id_need_replace)
                    return (feature_key,program.id)
                else:
                    logger.debug("Existing program better, no replacement in feature map", 
                                island_id=state.id,
                                feature_key=feature_key,
                                existing_program_id=program_id_need_replace,
                                candidate_program_id=program.id)
                    return None
        else:#若不需要替换 则直接添加
            logger.info("Adding new program to feature map", 
                       island_id=state.id,
                       feature_key=feature_key,
                       program_id=program.id)
            return (feature_key,program.id)




