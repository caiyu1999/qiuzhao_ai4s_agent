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
import logging
import asyncio
import os
logger = logging.getLogger(__name__)





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
        # #logger.info("node_init_status: 开始执行 execute")
        # 验证必要的配置参数
        if config.init_program_path == "":
            #logger.error("init_program is not set")
            raise ValueError("init_program is not set")
        if config.evalutor_file_path == "":
            #logger.error("evaluator_file_path is not set")
            raise ValueError("evaluator_file_path is not set")
        if config.island.num_islands <= 0:
            #logger.error("num_islands must be greater than 0")
            raise ValueError("num_islands must be greater than 0")

        # #logger.info("node_init_status: 配置参数验证通过")

        # 提取文件信息
        file_extension = os.path.splitext(config.init_program_path)[1]
        if not file_extension:
            file_extension = ".py"  # 默认扩展名

        # #logger.info(f"node_init_status: 文件扩展名为 {file_extension}")

        # 加载和处理初始程序
        code = load_initial_program(config.init_program_path)
        # #logger.info("node_init_status: 初始程序已加载")
        language = extract_code_language(code)
        # #logger.info(f"node_init_status: 检测到语言为 {language}")

        # 生成唯一ID
        id = str(uuid.uuid4())
        # #logger.info(f"node_init_status: 生成唯一ID {id}")
        
        # 对初始程序进行评估 
        init_program_code = code
        init_program_id = id
        evaluation_file = self.config.evalutor_file_path
        # #logger.info("node_init_status: 开始评估初始程序")
        eval_result,artifact_update = await self._evaluate_program(init_program_code,init_program_id,evaluation_file,state)
        # #logger.info(f"node_init_status: 初始程序评估完成，metrics: {eval_result.metrics}")
        artifact = None 
        artifacts_json = None
        artifact_dir = None
        if self.config.enable_artifacts: #如果开启工件 先生成工件 并存储工件
            # #logger.info("node_init_status: 启用工件存储，准备存储工件")
            artifact = eval_result.artifacts
            artifact.update(artifact_update) # 更新工件
            artifacts_json,artifact_dir = store_artifacts(init_program_id,artifact,state,self.config)
            # #logger.info(f"node_init_status: 工件已存储, artifact_dir: {artifact_dir}")
        
        # 生成program对象 
        # #logger.info("node_init_status: 生成初始Program对象")
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
        
    
        # 初始化岛屿相关数据结构
        island_id_list = [f"{i}" for i in range(config.island.num_islands)]
        islandstate_dict = {}
        for island_id in island_id_list:
            temp_Island_state = IslandState(id = island_id)
            temp_Island_state.programs.add_program(init_program.id,init_program)
            temp_Island_state.latest_program = init_program 
            temp_Island_state.status = IslandStatus.INIT_STATE
            temp_Island_state.all_programs.add_program(init_program.id,init_program)
            temp_Island_state.archive.add_program(init_program.id,init_program)
            temp_Island_state.language = language
            temp_Island_state.all_best_program = init_program
            temp_Island_state.next_meeting = self.next_meeting
            temp_Island_state.now_meeting = 0
            islandstate_dict[island_id] = temp_Island_state

            
        # #logger.info("node_init_status: 初始化岛屿相关数据结构")
        island_programs_ = Programs_container()
        island_programs_.add_program(init_program_id,init_program)
        all_programs = island_programs_.copy()
        archive = island_programs_.copy()
        feature_map = {} # 全部程序中的特征坐标 全局更新 e.g{"program_id":[0,1,2,-1]}
        num_islands = config.island.num_islands
        islands_id = island_id_list # 岛屿的id 全局唯一 不会被更新 安全
        best_program = init_program 
        
        
        return {
            "init_program":id,
            "best_program":code,
            "best_program_id":id,
            "best_metrics":eval_result.metrics,
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
        # #logger.info("node_init_status: __call__ invoked")
        # logger.info("node_init_status: 开始执行 execute")
        result = self.execute(state)
        
        # 发送消息到可视化服务器
        # try:
        #     import json
        #     client = SimpleClient(port=self.port)
        #     message = {
        #         "node_name": "init_status",
        #         "status": "completed",
        #         "data": {
        #             "num_islands": result.get("num_islands", 0) if isinstance(result, dict) else 0,
        #             "language": result.get("language", "unknown") if isinstance(result, dict) else "unknown"
        #         }
        #     }
        #     client.send_message(json.dumps(message))
        # except Exception as e:
        #     logger.warning(f"发送可视化消息失败: {e}")
        
        # logger.info(f"node_init_status: 执行结束")
        return result
    
    async def _llm_evaluate(self, program_code: str) -> Dict[str, Any]:
        """
        Use LLM to evaluate code quality

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging

        Returns:
            Dictionary of metric name to score
        """
        # #logger.info("node_init_status: 开始 LLM 评估")
        if not self.llm_evaluator:
            # #logger.warning("node_init_status: 未配置 LLM 评估器")
            return {}

        try:
            # Create prompt for LLMThreadSafePrograms
            prompt = self.prompt_sampler.build_prompt(
                current_program=program_code, template_key="evaluation"
            )

            # #logger.info(f"node_init_status: LLM prompt 构建完成: {prompt}")

            # Get LLM response
            responses = await self.llm_evaluator.invoke_parallel(
                prompt=prompt,
                structure=self.structure,
                key=self.key,
            )
            # #logger.info(f"node_init_status: LLM 响应: {responses}") 
            # LLM 响应: {'readability': [0.85], 'maintainability': [0.9], 'efficiency': [0.8], 'other_information': ['The code is well-structured, clear, and modular, making it maintainable and easy to modify. However, the nested loop in `compute_max_radii` could be optimized further, especially for larger n.']}

            try: # 当前版本返回的一般是一个dict {key:[value1,value2,value3]}

                artifacts = {}
                metrics = {}
                for key, value_list in responses.items():
                    if key == "other_information": # 其他信息 直接赋给artifacts
                        continue
                    # #logger.debug(f"node_init_status: LLM 响应 key: {key}")
                    length_ = len(value_list) #获取value_list的长度
                    metrics[key] = sum(value_list) / length_
                    
                artifacts['other_information'] = responses['other_information']

                # #logger.info(f"node_init_status: LLM 评估结果 metrics: {metrics}, artifacts: {artifacts}")
                return EvaluationResult(
                    metrics=metrics,
                    artifacts=artifacts,
                ).to_dict()

            except Exception as e:
                # #logger.warning(f"Error parsing LLM response: {str(e)}")
                return {}

        except Exception as e:
            logger.error(f"Error in node_init_status LLM evaluation: {str(e)}")
            #traceback.logger.info_exc()
            return {}
    def _process_evaluation_result(self, result: Any) -> EvaluationResult:
        """
        Process evaluation result to handle both dict and EvaluationResult returns

        Args:
            result: Raw result from evaluation function

        Returns:
            EvaluationResult instance
        """
        # #logger.info("node_init_status: 处理评估结果 _process_evaluation_result")
        if isinstance(result, dict):
            # Backward compatibility - wrap dict in EvaluationResult
            # #logger.debug("node_init_status: 评估结果为 dict，转换为 EvaluationResult")
            return EvaluationResult.from_dict(result)
        elif isinstance(result, EvaluationResult):
            # New format - use directly
            # #logger.debug("node_init_status: 评估结果为 EvaluationResult 实例")
            return result
        else:
            # Error case - return error metrics
            # #logger.warning(f"Unexpected evaluation result type: {type(result)}")
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
        # #logger.info(f"node_init_status: 开始评估程序 program_id={program_id}")
        start_time = time.time()
        program_id_str = f" {program_id}" if program_id else ""

        # Check if artifacts are enabled 是否开启工件通道
        artifacts_enabled = self.config.enable_artifacts

        # Retry logic for evaluation
        last_exception = None

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(program_code.encode("utf-8"))
            temp_file_path = temp_file.name
        # #logger.info(f"node_init_status: 临时文件创建于 {temp_file_path}")
            
        eval_result = EvaluationResult()
        artifact_update = {}
        
        try:
            # #logger.info("node_init_status: 开始执行评估函数")
            # Run evaluation
            if self.config.evaluator.cascade_evaluation:# 分级评估 会返回一个metrics和artifacts的EvaluationResult 工件在这里产生
                # #logger.info("node_init_status: 使用 cascade_evaluate")
                result = await cascade_evaluate(temp_file_path,evaluation_file,self.config)
            else:
                # #logger.info("node_init_status: 使用 direct_evaluate")
                result = await direct_evaluate(temp_file_path,evaluation_file,self.config)

            # Process the result based on type 
            eval_result = self._process_evaluation_result(result)
            # #logger.info(f"node_init_status: 评估函数返回结果: {eval_result.metrics}")

            # Check if this was a timeout and capture artifacts if enabled 
            if artifacts_enabled and program_id and eval_result.metrics.get("timeout") is True:
                # #logger.warning("node_init_status: 检测到评估超时，准备更新工件")
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
                # #logger.info("node_init_status: 启用 LLM 反馈评估")
                llm_result = await self._llm_evaluate(program_code) # 返回一个dict  {'readability': 0.85, 'maintainability': 0.8, 'efficiency': 0.7, 'other_information': ''}
                # #logger.info(f"node_init_status: LLM 反馈评估结果: {llm_result}")
                llm_eval_result = self._process_evaluation_result(llm_result)

                for name, value in llm_result.items():
                    eval_result.metrics[f"llm_{name}"] = value

            # Store artifacts if enabled and present
            if (
                artifacts_enabled
                and (
                    eval_result.has_artifacts()
                    or (llm_eval_result and llm_eval_result.has_artifacts())
                )
                and program_id
            ):
                # #logger.info("node_init_status: 工件存储条件满足，准备合并工件")
                if state.all_programs.get_program(program_id).artifacts_json is not None:
                    artifact_update = {}

                # Merge eval_result artifacts with llm artifacts if they exist
                if eval_result.has_artifacts():
                    artifact_update=eval_result.artifacts
                    # #logger.debug(
                    #     f"Program{program_id_str} returned artifacts: "
                    #     f"{eval_result.artifacts}"
                    # )

                if llm_eval_result and llm_eval_result.has_artifacts():
                    artifact_update=llm_eval_result.artifacts
                    # #logger.debug(
                    #     f"Program{program_id_str} returned LLM artifacts: "
                    #     f"{llm_eval_result.artifacts}"
                    # )

            elapsed = time.time() - start_time
            # #logger.info(
            #     f"Evaluated program{program_id_str} in {elapsed:.2f}s: "
            #     f"{format_metrics_safe(eval_result.metrics)}"
            # )
            # #logger.info(f"node_init_status: artifact_update: {artifact_update}")
            # Return just metrics for backward compatibility
            return eval_result , artifact_update

        except asyncio.TimeoutError: #如果是超时错误 
            # Handle timeout specially - don't retry, just return timeout result
            # #logger.warning(f"Evaluation timed out after {self.config.evaluator.timeout}s")

            # Capture timeout artifacts if enabled
            if artifacts_enabled and program_id:
                artifact_update = {
                    "timeout": True,
                    "timeout_duration": self.config.evaluator.timeout,
                    "failure_stage": "evaluation",
                    "error_type": "timeout",
                }
                eval_result.metrics.update({"error": 0.0,"timeout": True})

            # #logger.warning("node_init_status: 评估超时，返回超时结果")
            return eval_result , artifact_update

        except Exception as e:
            last_exception = e
            # #logger.warning(
            #     f"Evaluation attempt failed for program{program_id_str}: {str(e)}"
            # )
            

            # Capture failure artifacts if enabled
            if artifacts_enabled and program_id:
                artifact_update = {
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    "failure_stage": "evaluation",
                }
            # #logger.error("node_init_status: 评估发生异常，返回异常结果")

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                # #logger.info(f"node_init_status: 临时文件 {temp_file_path} 已删除")

        # All retries failed
        # #logger.error(
        #     f"All evaluation attempts failed for program{program_id_str}. Last error: {str(last_exception)}"
        # )
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

        #岛屿的并行评估
        try:
            current_program = state.latest_program.code
            current_program_id = state.latest_program.id
            evaluation_file = self.config.evalutor_file_path
            # #logger.info(f"开始评估岛屿{self.island_id}的程序: {current_program_id}")
            # #logger.info(f"程序的code为: {current_program}")
            # #logger.info(f"程序的id为: {current_program_id}")
            # #logger.info(f"程序的evaluation_file为: {evaluation_file}")

            eval_result,artifact_update = await self._evaluate_program(current_program,current_program_id,evaluation_file,state)
            # #logger.info(f"node_evaluate: 评估结果: {eval_result.to_dict()}")
            # #logger.info(f"node_evaluate: 工件更新: {artifact_update}")
            
            changes_summary = state.change_summary
            parent_id = state.sample_program.id
            parent_metrics = state.all_programs.get_program(parent_id).metrics
            
            artifact = None 
            artifacts_json = None
            artifact_dir = None
            if self.config.enable_artifacts: #如果开启工件 先生成工件 并存储工件
                artifact = eval_result.artifacts
                artifact.update(artifact_update) # 更新工件
                artifacts_json,artifact_dir = store_artifacts(current_program_id,artifact,state,self.config)
            #logger.info(f"node_evaluate: 工件存储: {artifacts_json}")
            #logger.info(f"node_evaluate: 工件存储: {artifact_dir}")
            
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
            
            return current_program
        except Exception as e:
            logger.error(f"node_evaluate: 评估发生异常: {e}")
            return None
    def __call__(self,state:IslandState):
        '''
        这个节点计算初始prgram的信息 并更新current_program
        '''
        logger.info(f"Island:{state.id} START node_evaluate")
        current_program = asyncio.run(self.execute_async(state))
        # logger.info(f"current_program:{current_program}")
        logger.info(f"Island:{state.id} END node_evaluate")
        if self.client is None:
            self.client = SimpleClient(self.config.port)
        update_dict = {}
        if current_program is None:
            logger.error(f"Island:{state.id} node_evaluate: 评估发生异常")
            #目前暂时进行下一轮
            update_dict = {
                "status":IslandStatus.EVALUATE_CHILD,
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
            "status":IslandStatus.EVALUATE_CHILD,
            "latest_program":current_program,
            "evaluate_success":True,
        }
        
        
        self.client.send_message({
                "island_id":state.id,
                "update_dict":update_dict,
                
            })
        
        return update_dict
        
        
    async def _llm_evaluate(self, program_code: str) -> Dict[str, Any]:
        """
        Use LLM to evaluate code quality

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging

        Returns:
            Dictionary of metric name to score
        """
        if not self.llm_evalutor:
            return {}

        try:
            # Create prompt for LLMThreadSafePrograms
            prompt = self.prompt_sampler.build_prompt(
                current_program=program_code, template_key="evaluation"
            )


            # Get LLM response
            responses = await self.llm_evalutor.invoke_parallel(
                prompt=prompt,
                structure=self.structure,
                key=self.key,
            )
            #logger.info(f"responses:{responses}")
            #logger.info(f"parallel generate success ")

            try: # 当前版本返回的一般是一个dict {key:[value1,value2,value3]}

                artifacts = {}
                metrics = {}
                # {'readability': [0.85], 'maintainability': [0.75], 'efficiency': [0.8], 'reasoning': ['The code is well-structured and uses clear function separation, making it understandable and maintainable. The use of numpy for array operations enhances performance, but the nested loops in the `compute_max_radii`
                # function could be optimized further. Overall, the constructor approach is explicit and avoids the complexity of iterative algorithms.']}
                for key, value_list in responses.items():
                    if key =="other_information": # 其他信息 直接赋给artifacts
                        continue
                    #logger.info(f"key:{key}")
                    length_ = len(value_list) #获取value_list的长度
                    metrics[key] = sum(value_list) / length_
                ##import pdb;pdb.set_trace()
                artifacts['other_information'] = responses['other_information']
                #logger.info(f"artifacts in node_evaluate:{artifacts}")
                return EvaluationResult(
                    metrics=metrics,
                    artifacts=artifacts,
                ).to_dict()

            except Exception as e:
                #logger.warning(f"Error parsing LLM response: {str(e)}")
                return {}

        except Exception as e:
            #logger.error(f"Error in LLM evaluation: {str(e)}")
            traceback.logger.info_exc()
            return {}
    def _process_evaluation_result(self, result: Any) -> EvaluationResult:
        """
        Process evaluation result to handle both dict and EvaluationResult returns

        Args:
            result: Raw result from evaluation function

        Returns:
            EvaluationResult instance
        """
        if isinstance(result, dict):
            # Backward compatibility - wrap dict in EvaluationResult
            return EvaluationResult.from_dict(result)
        elif isinstance(result, EvaluationResult):
            # New format - use directly
            return result
        else:
            # Error case - return error metrics
            #logger.warning(f"Unexpected evaluation result type: {type(result)}")
            return EvaluationResult(metrics={"error": 0.0})

    async def _evaluate_program(
        self,
        program_code,
        program_id,
        evaluation_file,
        state:IslandState,
    ) -> Tuple[EvaluationResult, Dict[str, Any]]:

        start_time = time.time()
        program_id_str = f" {program_id}" if program_id else ""

        # Check if artifacts are enabled 是否开启工件通道
        artifacts_enabled = self.config.enable_artifacts

        # Retry logic for evaluation
        last_exception = None

        # for attempt in range(self.config.evaluator.max_retries + 1):
        #     # Create a temporary file for the program
        
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(program_code.encode("utf-8"))
            temp_file_path = temp_file.name
            #logger.info(f"评估中,temp_file_path为: {temp_file_path}")
            
        eval_result = EvaluationResult()
        artifact_update = {}
        
        try:
            
            # Run evaluation
            if self.config.evaluator.cascade_evaluation:# 分级评估 会返回一个metrics和artifacts的EvaluationResult 工件在这里产生
                # Run cascade evaluation
                result = await cascade_evaluate(temp_file_path,evaluation_file,self.config)
            else:
                # Run direct evaluation
                result = await direct_evaluate(temp_file_path,evaluation_file,self.config)


            # 如果报错 result中会包含报错的信息
            
            # Process the result based on type 
            eval_result = self._process_evaluation_result(result)
            #logger.info(f"评估结果: {eval_result.to_dict()}")
            # Check if this was a timeout and capture artifacts if enabled 
            # 包含工件 但是超时
            if artifacts_enabled and program_id and eval_result.metrics.get("timeout") is True:
                #logger.info(f"评估超时,包含artifacts 但是timeout为True")
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
                #logger.info(f"开始LLM评估")
                # llm_result = asyncio.run(self._llm_evaluate(program_code)) # 返回一个dict 
                llm_result = await self._llm_evaluate(program_code)
                # 若llm_result为{} 则证明llm评估失败
                #logger.info(f"LLM评估结果:{llm_result}")
                llm_eval_result = self._process_evaluation_result(llm_result)
                #logger.info(f"LLM评估结果(after process): {llm_eval_result.to_dict()}")
                for name, value in llm_result.items():
                    eval_result.metrics[f"llm_{name}"] = value
                ##import pdb;pdb.set_trace()



            # Store artifacts if enabled and present
            if (
                artifacts_enabled
                and (
                    eval_result.has_artifacts()
                    or (llm_eval_result and llm_eval_result.has_artifacts())
                )
                and program_id
            ):
                if state.all_programs.get_program(program_id).artifacts_json is not None:
                    artifact_update = {}

                # Merge eval_result artifacts with llm artifacts if they exist
                if eval_result.has_artifacts():
                    artifact_update=eval_result.artifacts
                    #logger.debug(
                        # f"Program{program_id_str} returned artifacts: "
                        # f"{eval_result.artifacts}"
                    # )

                if llm_eval_result and llm_eval_result.has_artifacts():
                    artifact_update=llm_eval_result.artifacts
                    #logger.debug(
                        # f"Program{program_id_str} returned LLM artifacts: "
                        # f"{llm_eval_result.artifacts}"
                    # )

            elapsed = time.time() - start_time
            #logger.info(
                # f"Evaluated program{program_id_str} in {elapsed:.2f}s: "
                # f"{format_metrics_safe(eval_result.metrics)}"
            # )
            #logger.info(f"artifact_update:{artifact_update}")
            # Return just metrics for backward compatibility
            return eval_result , artifact_update

        except asyncio.TimeoutError: #如果是超时错误 
            # Handle timeout specially - don't retry, just return timeout result
            #logger.warning(f"Evaluation timed out after {self.config.evaluator.timeout}s")

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
            #logger.warning(
            #     f"Evaluation attempt failed for program{program_id_str}: {str(e)}"
            # )
            traceback.logger.info_exc()

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
    # All retries failed
        #logger.error(
        #     f"All evaluation attempts failed for program{program_id_str}. Last error: {str(last_exception)}"
        # )
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
        # state_island:IslandState = getattr(state, self.island_id)
        try:
            
            parent_program = self._sample_parent(state)
            if parent_program is None:
                #logger.error(f"采样父代程序失败: 返回None")
                return None,[]
            
            parent_id = parent_program.id
            #logger.info(f"采样父代成功 父代ID:{parent.id}")
            
        except Exception as e:
            #logger.error(f"采样父代程序失败: {e}")s
            return None,[]

        inspirations = self._sample_inspirations(state,parent_id)
        logger.info(f"采样灵感程序成功 灵感程序ID:{inspirations}")
        return parent_program,inspirations

    def __call__(self, state: IslandState, config: Optional[Any] = None) -> Dict[str, Any]:

        # 执行采样逻辑
        # logger.info(f"岛屿{state.IslandState.id}的node_sample_parent_inspiration开始")
        logger.info(f"Island:{state.id} START node_sample_parent_inspiration")
        parent_program,inspirations = self.execute(state)
        # state_island:IslandState = getattr(state, self.island_id)
        # state_island.sample_program = parent_program
        # state_island.sample_inspirations = inspirations
        # state_island.status = IslandStatus.SAMPLE
        logger.info(f"Island:{state.id} END node_sample_parent_inspiration")
        # 只返回需要更新的字段，让LangGraph的reducer处理并发
        if self.client is None:
            self.client = SimpleClient(self.config.port)
        update_dict = {
            "sample_program":parent_program,
            "sample_inspirations":inspirations,
            "status":IslandStatus.SAMPLE
        }
        self.client.send_message({
                "island_id":state.id,
                "update_dict":update_dict,
                
            })
        return update_dict

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
        inspirations = []

        #若最优程序存在 且与父代不同 且在所有的程序中 则加入灵感程序
        if (
            state.all_best_program is not None
            and state.all_best_program.id != parent_id
            and state.all_best_program.id in state.all_programs.get_program_ids()
        ):

            inspirations.append(state.all_best_program.id)
            #logger.info(f"在灵感程序候选列表中加入顶级程序 inspiration:{inspirations}")

        # 添加顶级程序作为灵感
        top_n = max(1, int(self.n * self.config.island.elite_selection_ratio))
        top_programs = get_top_programs(state,n=top_n,metric=self.metric)
        for program in top_programs:
            if program.id not in inspirations and program.id != parent_id:
                inspirations.append(program.id)
                #logger.info(f"在灵感程序候选列表中加入顶级程序 inspiration:{inspirations}")

        # 添加多样性程序
        if len(state.all_programs) > self.n and len(inspirations) < self.n:
            # 计算要添加的多样性程序数量（最多到剩余位置）
            remaining_slots = self.n - len(inspirations)


            # 从不同的特征格子采样以获得多样性

            feature_coords = _calculate_feature_coords(self.config,state,state.all_programs.get_program(parent_id))#这里获得的是一个描述父代多样性的坐标
            #logger.info(f"计算多样性坐标 多样性坐标:{feature_coords}")
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
                        #logger.info(f"在灵感程序候选列表中加入多样性程序 inspiration:{inspirations}")
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
                    # random_programs = [state.all_programs.get_program(pid) for pid in random_ids]
                    # nearby_programs.extend(random_programs)
                    inspirations.extend(random_ids)
                    #logger.info(f"在灵感程序候选列表中加入随机程序 inspiration:{inspirations}")
            inspirations.extend(nearby_programs)
            
        # 保持顺序去重 inspirations
        seen = set()
        inspirations = [x for x in inspirations if not (x in seen or seen.add(x))]
        #logger.info(f"去重后的灵感程序列表 inspiration:{inspirations}")
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
        # 使用探索比例和利用比例决定采样策略
        rand_val = random.random()

        if rand_val < self.config.island.exploration_ratio:
            #logger.info("node_sample_parent_inspiration: 使用探索性采样")
            # 探索：从当前岛屿采样（多样性采样）
            return self._sample_exploration_parent(state)
        elif rand_val < self.config.island.exploration_ratio + self.config.island.exploitation_ratio:
            #logger.info("node_sample_parent_inspiration: 使用利用性采样")
            # 利用：从归档采样（精英程序）
            return self._sample_exploitation_parent(state)
        else:
            #logger.info("node_sample_parent_inspiration: 使用随机性采样")
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
        # 获取当前岛屿的程序ID列表
        # current_island索引对应当前子图处理的岛屿
        current_island_programs = state.programs.get_program_ids()

        # 岛屿完整性验证
        # 正常情况下每个岛屿都应该包含至少一个初始程序
        if not current_island_programs:
            raise ValueError(
                f"岛屿 {state.id} 未正确初始化，程序列表为空。"
                f"请确保初始程序已正确设置到所有岛屿中。"
            )

        # 探索性随机采样
        # 使用random.choice确保每个程序被选中的概率相等
        parent_id = random.choice(list(current_island_programs))

        # 从全局程序字典中获取完整的Program对象
        # 这里假设程序ID的一致性已经在其他地方得到保证

        return state.all_programs.get_program(parent_id)

    def _sample_exploitation_parent(self, state: IslandState) -> Optional[Program | None | Any]:
        """
        利用性采样父代程序（从精英归档采样策略）

        该方法实现了岛屿模型中的利用性采样策略，用于从精英归档中
        选择一个程序作为下一轮进化的父代程序。这种采样方式有助于
        利用已有的优秀程序，加速收敛到全局最优解。

        算法流程：
        """

        archive_programs_ids = state.archive.get_program_ids() # 精英归档的id集合 archive在初始化时候会被初始化 
        #logger.info(f"node_sample_parent_inspiration: 精英归档程序: {archive_programs_ids}")
        # 尽可能获得当前岛屿内部的精英归档
        archive_in_current_island = [pid for pid in archive_programs_ids if pid in state.programs.get_program_ids()]
        #logger.info(f"node_sample_parent_inspiration: 当前岛屿{state.id}的精英归档程序: {archive_in_current_island}")
        #优先从自己的岛屿上采样 如果自己的岛屿上没有精英归档类别的程序 就随机采样

        if len(archive_in_current_island) > 0: #如果自己的岛屿上有精英归档类别的程序 则从自己的岛屿上采样
            #logger.info("node_sample_parent_inspiration: 从自己的岛屿上采样")
            return state.all_programs.get_program(random.choice(archive_in_current_island))
        else: #如果自己的岛屿上没有精英归档类别的程序 则从所有岛屿的精英归档中采样
            if len(archive_programs_ids) > 0:
                #logger.info("node_sample_parent_inspiration: 从所有岛屿的精英归档中采样")
                return state.all_programs.get_program(random.choice(archive_programs_ids))
            else:
                #logger.warning("node_sample_parent_inspiration: 没有精英归档程序，使用随机性采样")
                return self._sample_random_parent(state)

    def _sample_random_parent(self,state:IslandState) -> Optional[Program | None | Any]:
        """
        完全随机采样父代程序

        Returns:
            Program: 选中的父代程序
        """
        if not state.all_programs:
            raise ValueError("No programs available for sampling")

        # 从自己岛屿上的所有程序中随机采样
        current_island_programs = state.programs.get_program_ids()
        if not current_island_programs:
            raise ValueError(f"当前岛屿 {state.id} 的程序列表为空")
        program_id = random.choice(list(current_island_programs))
        return state.all_programs.get_program(program_id)

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
        #logger.info(f"岛屿{self.island_id}的构建prompt开始")
        return self._build_prompt(state)
    def __call__(self, state: IslandState, config: Optional[Any] = None) -> Dict[str, Any]:
        logger.info(f"Island:{state.id} START node_build_prompt")
        prompt = self.execute(state)
        logger.info(f"Island:{state.id} END node_build_prompt")
        if self.client is None:
            self.client = SimpleClient(self.config.port)
        update_dict = {
            "prompt": prompt,
            "status": IslandStatus.BUILD_PROMPT
        }
        self.client.send_message({
                "island_id":state.id,
                "update_dict":update_dict,
                
            })
        return update_dict
    def _build_prompt(self,state:IslandState)->str:

        # 若当前program为空 则使用父代程序
        current_program_id = state.latest_program.id if state.latest_program.id is not None else state.sample_program.id
        current_program = state.all_programs.get(current_program_id)
        
        if current_program is None:
            #logger.error(f"当前程序为空: {current_program_id}")
            return ""

        parent_program = state.all_programs.get(state.sample_program.id)
        if parent_program is None:
            #logger.error(f"父代程序为空: {state.sample_program.id}")
            return ""
        
        parent_code = parent_program.code
        parent_metrics = parent_program.metrics
        #logger.info(f"岛屿{self.island_id}的父代程序metrics: {parent_program.to_dict()}")

        previous_programs = []
        #当前的父代程序

        for _ in range(3):
            #如果父代程序有父代程序 则将父代程序的父代程序加入previous_programs 否则结束循环
            if isinstance(parent_program,Program) and parent_program.parent_id:
                previous_programs.append(state.all_programs.get(parent_program.parent_id))
                parent_program = state.all_programs.get(parent_program.parent_id)
            
            else:
                break
        # if previous_programs != []:
        #     # previous_programs = [state.all_programs.get_program(pid).to_dict() for pid in previous_programs if pid is not None]
        #     #logger.info(f"岛屿{self.island_id}的previous_programs: {previous_programs}")
        # else:
        #     previous_programs = []



        inspirations = state.sample_inspirations
        
        inspirations_programs = [state.all_programs.get_program(ins).to_dict() for ins in inspirations if ins is not None]
        
        #logger.info(f"岛屿{self.island_id}在build_prompt中的inspirations: {inspirations}")

        top_programs = [i.to_dict() for i in get_top_programs(state,n=5,metric=self.metric)]
        
        #logger.info(f"岛屿{self.island_id}在build_prompt中的top_programs: {top_programs}")
        
        return self.prompt_sampler.build_prompt(
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
            #logger.info(f"岛屿{self.island_id}的LLM生成使用基于差异的演化")
        else:
            self.structure = ResponseFormatter_template_rewrite
            self.key = ["suggestion","rewrite_code"]
            #logger.info(f"岛屿{self.island_id}的LLM生成使用基于重写的演化")
        self.client =  None

    def _llm_generate(self,state:IslandState):
        return self.llm.invoke(state.prompt,self.structure,self.key)
    def execute(self,state:IslandState):
        
        llm_response:Optional[Dict[str,Any]|None] = self._llm_generate(state)
        # logger.info(f"岛屿{self.island_id}的LLM生成结果: {llm_response}")
        #logger.info(f"岛屿{self.island_id}的LLM生成结果: {llm_response}")
        # logger.info(f"岛屿{self.island_id}的LLM生成成功")
        generation = state.iteration
        if self.client is None:
            self.client = SimpleClient(self.config.port)
        if llm_response is None:
            # 失败 
            #logger.info(f"岛屿{self.island_id}在llm_generate的过程中的response为None")
            update_dict = {
                "status": IslandStatus.LLM_GENERATE,
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
            #logger.info(f"岛屿{self.island_id}使用基于diff的进化方式")
            parent_program = state.all_programs.get_program(state.sample_program.id)
            if parent_program is None:
                #logger.error(f"父代程序为空: {state.sample_program.id}")
                update_dict = {
                    # "status": (self.island_id,IslandStatus.LLM_GENERATE),
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
            diff_blocks = extract_diffs(diff_code)
            if not diff_blocks or diff_blocks == []:
                #logger.warning(f"{state.status[self.island_id]}:岛屿{self.island_id}第{state.generation_count[self.island_id]}轮LLM的输出没有diff_blocks")
                update_dict = {
                    # "status": (self.island_id,IslandStatus.LLM_GENERATE),
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
            child_code = apply_diff(parent_code,diff_code)
            change_summary = format_diff_summary(diff_blocks)
            #生成子代的id
            child_id = str(uuid.uuid4())
            #logger.info(f"成功生成岛屿{self.island_id}的子代id: {child_id}")
            
            child_Program = Program(id=child_id,
                                    code=child_code,
                                    parent_id=parent_program.id,
                                    )

            update_dict = {
                "diff_message": diff_code,
                "suggestion_message": suggestion,
                "status": IslandStatus.LLM_GENERATE,
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
            rewrite_code = llm_response["rewrite_code"]
            if rewrite_code is None or rewrite_code == "":
                #logger.warning(f"{state.status[self.island_id]}:岛屿{self.island_id}第{state.generation_count[self.island_id]}轮LLM的输出没有rewrite_code")
                update_dict = {
                    "status": IslandStatus.LLM_GENERATE,
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
            new_code = parse_full_rewrite(rewrite_code,state.language)
            if not new_code:
                #logger.warning(f"{state.status[self.island_id]}:岛屿{self.island_id}第{state.generation_count[self.island_id]}轮LLM的输出没有new_code")
                return {
                        "status": IslandStatus.LLM_GENERATE,
                        "llm_generate_success":False,
                        "iteration":generation+1,#重新采样 代数+1
                        "now_meeting":state.now_meeting+1,
                        "next_meeting":state.next_meeting-1,
                    }
            change_summary = "full rewrite"
            child_id = str(uuid.uuid4())
            #logger.info(f"成功生成岛屿{self.island_id}的子代id: {child_id}")
            child_program = Program(id=child_id,code=new_code,parent_id=state.sample_program.id)
            update_dict = {
                "rewrite_message":rewrite_code,
                "suggestion_message":llm_response["suggestion"],
                "status": IslandStatus.LLM_GENERATE,
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
        logger.info(f"Island:{state.id} START node_llm_generate")
        result = self.execute(state)
        logger.info(f"Island:{state.id} END node_llm_generate")
        return result
    
    
    
    
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
        logger.info(f"Island:{state.id} START node_update")
        update_dict = self.execute(state)
        logger.info(f"Island:{state.id} END node_update")
        if self.client is None:
            self.client = SimpleClient(self.config.port)
        self.client.send_message({
                "island_id":state.id,
                "update_dict":update_dict,
                
            })
        return update_dict
    
    def execute(self,state:IslandState):
        
        # 此时all_programs island_program 中还没有新的program 
        current_program = state.latest_program
        
        # 与所有程序的副本相比 更好的程序
        best_program = self._best_program_update(state) 
        
       
        # 与当前岛屿上的程序相比 更好的程序
        best_program_each_island = self._best_program_this_island(state)
        
        
        island_programs_update = self._update_island_programs(state)
        
        
        ##import pdb;pdb.set_trace()
        # 是否更新all_programs 取决于island_programs 
        all_programs_update = None 
        if island_programs_update is None:
            all_programs_update = None 
        elif isinstance(island_programs_update,tuple):
            all_programs_update = island_programs_update
        else:
            raise ValueError("island_programs_update must be a tuple of length 2 or 4, but you give me a {}".format(len(island_programs_update)))

        newest_program_update = self._update_newest_program(state)
        archive_update = self._update_archive(current_program,state)
        feature_map_update = self._update_feature_map(current_program,state)
        
        update_dict = {}
        if best_program is not None : 
            update_dict["all_best_program"] = best_program
            
        if best_program_each_island is not None:
            update_dict["best_program"] = best_program_each_island
        
        if island_programs_update is not None:
            update_dict["programs"] = island_programs_update # e.g ("add",self.island_id,current_program)
       
        update_dict["latest_program"] = newest_program_update
        if archive_update is not None:
            update_dict["archive"] = archive_update
            
        if feature_map_update is not None:
            update_dict["feature_map"] = feature_map_update
        
        if all_programs_update is not None:
            update_dict["all_programs"] = all_programs_update
        
        current_iteration = state.iteration
        
        update_dict["iteration"] = current_iteration + 1
        update_dict["now_meeting"] = state.now_meeting + 1
        update_dict["next_meeting"] = state.next_meeting - 1
        
        update_dict["status"] = IslandStatus.UPDATE
        # for k,v in update_dict.items():
        #     logger.info(f"key:{k},value type:{type(v)}")
        ##import pdb;pdb.set_trace()
        
        return update_dict
    def _best_program_update(self,state:IslandState)->Optional[Program]:
        '''
        如果当前程序更好 则返回需要更新的 best_program 和 best_program_id 和 best_metrics
        '''
        
        current_program = state.latest_program
        
        best_program = state.all_best_program
        
        if _is_better(current_program,best_program) or best_program is None:#若当前程序更好 或者best_program为空
            best_program = current_program
            # best_program_id = current_program.id
            # best_metrics = current_program.metrics
            #logger.info(f"🌟 在岛屿{self.island_id}第 {state.generation_count[self.island_id]} 次迭代发现新的最优解: {current_program.id}")
            #logger.info(f"新指标: {format_metrics_safe(current_program.metrics)}")
            return best_program
        else:
            return None
    def _best_program_this_island(self,state:IslandState)->Optional[Program|None]:
        '''
        获取当前岛屿上最好的程序 并比较 如果当前程序更好 则更新best_program_each_island中[self.island_id]的id
        '''
        current_program = state.latest_program
        best_program_this_island = state.best_program
        # best_program_this_island = state.all_programs.get_program(best_program_each_island_id)
        ##import pdb;pdb.set_trace()
        if _is_better(current_program,best_program_this_island) or best_program_this_island is None:
            best_program_this_island = current_program
            return best_program_this_island
        else:
            return None
        
        
        
    def _update_island_programs(self,state:IslandState)->Optional[tuple[str, str, Program | Any | None]|Tuple[str,Program]|None]:
        #更新当前岛屿上的程序 这时候首先要检查岛屿上的程序个数是否超过max_island_programs_size
        max_island_programs_size = self.config.island.population_size
        # current_program_id = state.current_program_id[self.island_id]
        current_program = state.latest_program
        if len(state.programs) < max_island_programs_size:
            return ("add",current_program)
        else:
            # 如果岛屿上的程序个数到达了峰值 需要移除某个程序
            # 当前的程序会和岛屿中最差的程序进行对比 如果当前程序更好 则移除最差的程序 并添加当前程序 否则不进行任何操作 
            island_programs = state.programs.get_all_programs() # 获取岛屿上的所有程序 Dict{program_id:Program}
            metrics_dict = {pid:program.metrics for pid,program in island_programs.items()}
            worst_program_id = min(metrics_dict,key=lambda x:safe_numeric_average(metrics_dict[x]))
            worst_program = state.programs.get_program(worst_program_id)
            if _is_better(current_program,worst_program):
                return ("replace",worst_program_id,current_program)
            else:
                return None
    def _update_newest_program(self,state:IslandState):
        #更新当前岛屿上的最新程序 
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
        #检查归档是否已满，未满则直接添加
        if len(state.archive) < self.config.archive_size:
            
            return ("add",program)
        
        elif len(state.archive) == self.config.archive_size: #已满 将其与最差的程序对比 并决定是否替换最差的程序
            archive_programs = state.archive.get_all_programs()
            metrics_dict = {pid:program.metrics for pid,program in archive_programs.items()}
            worst_program_id = min(metrics_dict,key=lambda x:safe_numeric_average(metrics_dict[x]))
            worst_program = archive_programs[worst_program_id]
            if _is_better(program,worst_program):
                return ("replace",worst_program_id,program)
            else:
                return None
    def _update_feature_map(self,program:Program,state:IslandState):
        #更新特征网格
        #计算当前程序在特征网格中的坐标
        feature_coords = _calculate_feature_coords(self.config,state,program)

        # 更新MAP-Elites特征网格
        feature_key = _feature_coords_to_key(feature_coords)
        
        should_replace = feature_key  in state.feature_map #若feature_key在feature_map中，则需要替换
        
        if should_replace:#若需要替换 取出feature_map中对应的program_id
            program_id_need_replace = state.feature_map[feature_key] #旧的
            if program_id_need_replace not in state.all_programs: #如果旧的程序id不在all_programs中 则直接替换
                return (feature_key,program.id)
            else:
                if _is_better(program,state.all_programs.get_program(program_id_need_replace)):
                    return (feature_key,program.id)
                else:
                    return None
        else:#若不需要替换 则直接添加
            return (feature_key,program.id)




