import os 
import json 
from typing import Dict
import uuid
import tempfile
import traceback
from pydantic import BaseModel
# from openevolve_graph.Graph.Graph_Node_ABC import NodeType
from openevolve_graph.Graph.Graph_Node_ABC import NodeResult,SyncNode
from openevolve_graph.Graph.Graph_state import GraphState, IslandStatus 
from openevolve_graph.Config import Config
from openevolve_graph.program import Program
from openevolve_graph.utils.utils import (
    parse_full_rewrite, 
    safe_numeric_average,
    _calculate_feature_coords,
    _feature_coords_to_key,
    format_metrics_safe,
    _get_artifact_size,
    _artifact_serializer,
    store_artifacts
    )
from openevolve_graph.Prompt.sampler import PromptSampler_langchain
from openevolve_graph.utils.utils import (
    load_initial_program, 
    extract_code_language,
    apply_diff,extract_diffs,
    format_diff_summary,
    parse_full_rewrite
    )
from openevolve_graph.utils.thread_safe_programs import ThreadSafePrograms
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



class node_evaluate(SyncNode):
    '''
    评估待评估程序(current_program) 并更新state
    这个节点在未设定island_id时 会评估init program 并更新all_programs island_programs  
    在设定island_id时 会评估岛屿上的程序 并更新all_programs island_programs 
    '''
    def __init__(self,config:Config,island_id: Optional[str|None]):
        self.config = config
        self.island_id = island_id
        self.llm_evalutor = LLMs_evalutor(config)
        self.prompt_sampler = PromptSampler_langchain(config=config.prompt)
        if self.config.evaluator.use_llm_feedback: #如果是基于差异的演化 那么使用
            self.structure = ResponseFormatter_template_evaluator
            self.key = ["readability","maintainability","efficiency"]
        else:
            self.structure = None
            self.key = None
        
    def execute(self,state:GraphState):
        #从state中获取当前状态 是init评估还是岛屿的并行评估
        if self.island_id is None:
            #init评估 这些参数要赋给每一个岛屿
            print(state.current_program_id)
            current_program = state.current_program_code['0']
            current_program_id = state.current_program_id['0']
            evaluation_file = self.config.evalutor_file_path
            metrics,artifact = asyncio.run(self._evaluate_program(current_program,current_program_id,evaluation_file,state))
            current_program = Program(
                    id=current_program_id,
                    code=current_program,
                    language=state.language,
                    parent_id=None,
                    generation=0,
                    timestamp=time.time(),
                    metrics=metrics,
                    )
        else:
            #岛屿的并行评估
            current_program = state.current_program_code[self.island_id]
            current_program_id = state.current_program_id[self.island_id]
            evaluation_file = self.config.evalutor_file_path
            metrics,artifact = asyncio.run(self._evaluate_program(current_program,current_program_id,evaluation_file,state))
            changes_summary = state.llm_change_summary[self.island_id]
            parent_id = state.sample_program_id[self.island_id]
            parent_metrics = state.all_programs.get_program(parent_id).metrics
            if self.config.enable_artifacts:
                artifacts_json,artifact_dir = store_artifacts(current_program_id,artifact,state,self.config)
            current_program = Program(
                    id=current_program_id,
                    code=current_program,
                    language=state.language,
                    parent_id=state.sample_program_id[self.island_id],
                    generation=state.generation_count[self.island_id],
                    timestamp=time.time(),
                    metrics=metrics,
                    metadata={
                        "changes": changes_summary,
                        "parent_metrics": parent_metrics,
                    }, 
                    artifacts_json=artifacts_json,
                    artifact_dir=artifact_dir,
                    )
        return current_program
    def __call__(self,state:GraphState):
        '''
        这个节点计算初始prgram的信息 并将其添加到all_programs ,island_programs archive中
        '''
        current_program = self.execute(state)
        
        
        if self.island_id is None: #此时为初始状态
            
            # print("().add_program(current_program.id,current_program)","\n",ThreadSafePrograms().add_program(current_program.id,current_program))
            container = ThreadSafePrograms()
            container.add_program(current_program.id,current_program)
            return {
                "status":[(f"{island_id}",IslandStatus.INIT_EVALUATE) for island_id in state.islands_id],
                "all_programs":container,
                "island_programs":{island_id:container for island_id in state.islands_id},
            }
    
    
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
            
            print("prompt","\n",prompt)

            # Get LLM response
            responses = await self.llm_evalutor.invoke_parallel(
                prompt=prompt,
                structure=self.structure,
                key=self.key,
            )
            print("responses","\n",responses)
            # print(responses)
            
            

            # Extract JSON from response
            try: # 当前版本返回的一般是一个dict {key:[value1,value2,value3]}
                # Try to find JSON block
                # json_pattern = r"```json\n(.*?)\n```"
                # import re
         
                artifacts = {}
                metrics = {}
                # {'readability': [0.85], 'maintainability': [0.75], 'efficiency': [0.8], 'reasoning': ['The code is well-structured and uses clear function separation, making it understandable and maintainable. The use of numpy for array operations enhances performance, but the nested loops in the `compute_max_radii` 
                # function could be optimized further. Overall, the constructor approach is explicit and avoids the complexity of iterative algorithms.']}
                for key, value_list in responses.items():
                    print("key","\n",key)
                    length_ = len(value_list) #获取value_list的长度
                    metrics[key] = sum(value_list) / length_ 
                    for value in enumerate(value_list):# 遍历每一个value 若不是 取平均值赋给metrics[key]
                        if not isinstance(value, (int, float)): 
                            artifacts[key] = value
                            
                            
                # print(EvaluationResult(
                #     metrics=avg_metrics,
                #     artifacts=artifacts,
                # ))
                return EvaluationResult(
                    metrics=metrics,
                    artifacts=artifacts,
                ).to_dict()

            except Exception as e:
                logger.warning(f"Error parsing LLM response: {str(e)}")
                return {}

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
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
        if isinstance(result, dict):
            # Backward compatibility - wrap dict in EvaluationResult
            return EvaluationResult.from_dict(result)
        elif isinstance(result, EvaluationResult):
            # New format - use directly
            return result
        else:
            # Error case - return error metrics
            logger.warning(f"Unexpected evaluation result type: {type(result)}")
            return EvaluationResult(metrics={"error": 0.0})
        
    async def _evaluate_program(
        self,
        program_code,
        program_id,
        evaluation_file,
        state:GraphState,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
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

        # Check if artifacts are enabled 是否开启工件通道
        artifacts_enabled = self.config.enable_artifacts

        # Retry logic for evaluation
        last_exception = None
        
        # for attempt in range(self.config.evaluator.max_retries + 1):
        #     # Create a temporary file for the program
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            temp_file.write(program_code.encode("utf-8"))
            temp_file_path = temp_file.name

        try:
            artifact = {}
            # Run evaluation
            if self.config.evaluator.cascade_evaluation:# 分级评估
                # Run cascade evaluation
                result = await cascade_evaluate(temp_file_path,evaluation_file,self.config)
            else:
                # Run direct evaluation
                result = await direct_evaluate(temp_file_path,evaluation_file,self.config)

            # Process the result based on type
            eval_result = self._process_evaluation_result(result)

            # Check if this was a timeout and capture artifacts if enabled
            if artifacts_enabled and program_id and eval_result.metrics.get("timeout") is True:
                if program_id not in state.artifacts:
                    artifact = {}

                artifact={ # 注意这里是要更新的 而不是新建
                        "timeout": True,
                        "timeout_duration": self.config.evaluator.timeout,
                        "failure_stage": "evaluation",
                        "error_type": "timeout",
                    }
                

            # Add LLM feedback if configured
            llm_eval_result = None
            if self.config.evaluator.use_llm_feedback and self.llm_evalutor:
                llm_result = await self._llm_evaluate(program_code) # 返回一个dict  {'readability': 0.85, 'maintainability': 0.8, 'efficiency': 0.7}
                print("llm_result","\n",llm_result)
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
                if program_id not in state.artifacts:
                    artifact = {}

                # Merge eval_result artifacts with llm artifacts if they exist
                if eval_result.has_artifacts():
                    artifact=eval_result.artifacts
                    logger.debug(
                        f"Program{program_id_str} returned artifacts: "
                        f"{eval_result.artifacts}"
                    )

                if llm_eval_result and llm_eval_result.has_artifacts():
                    artifact=llm_eval_result.artifacts
                    logger.debug(
                        f"Program{program_id_str} returned LLM artifacts: "
                        f"{llm_eval_result.artifacts}"
                    )

            elapsed = time.time() - start_time
            logger.info(
                f"Evaluated program{program_id_str} in {elapsed:.2f}s: "
                f"{format_metrics_safe(eval_result.metrics)}"
            )
            print("artifact","\n",artifact)
            # Return just metrics for backward compatibility
            return eval_result.metrics , artifact 

        except asyncio.TimeoutError:
            # Handle timeout specially - don't retry, just return timeout result
            logger.warning(f"Evaluation timed out after {self.config.evaluator.timeout}s")

            # Capture timeout artifacts if enabled
            if artifacts_enabled and program_id:
                artifact = {
                    "timeout": True,
                    "timeout_duration": self.config.evaluator.timeout,
                    "failure_stage": "evaluation",
                    "error_type": "timeout",
                }
            
            return {"error": 0.0, "timeout": True},artifact

        except Exception as e:
            last_exception = e
            logger.warning(
                f"Evaluation attempt failed for program{program_id_str}: {str(e)}"
            )
            traceback.print_exc()

            # Capture failure artifacts if enabled
            if artifacts_enabled and program_id:
                artifact = {
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    "failure_stage": "evaluation",
                }

            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    # All retries failed
        logger.error(
            f"All evaluation attempts failed for program{program_id_str}. Last error: {str(last_exception)}"
        )
        return {"error": 0.0} , artifact
    
    
    

class node_defer(SyncNode):
    def __init__(self,config:Config):
        self.config = config
        
    def execute(self,state:GraphState):
        return state
    
    def __call__(self,state:GraphState):
        return state

class node_init_status(SyncNode):
    '''
    初始化图的状态 将init program的id和code放入current program id ,current program code中  
    但是并不更新 all_programs archive island_programs 
    这些内容会在node_evaluate中更新
    '''
    
    def __init__(self,config:Config):
        self.config = config
        self.num_islands = config.island.num_islands
        
    def execute(self, state: BaseModel) -> NodeResult | Dict[str, Any]:
        config = self.config
        # 验证必要的配置参数
        if config.init_program_path == "":
            raise ValueError("init_program is not set")
        if config.evalutor_file_path == "":
            raise ValueError("evaluator_file_path is not set")
        if config.island.num_islands <= 0:
            raise ValueError("num_islands must be greater than 0")

        # 提取文件信息
        file_extension = os.path.splitext(config.init_program_path)[1]
        if not file_extension:
            file_extension = ".py"  # 默认扩展名

        # 加载和处理初始程序
        code = load_initial_program(config.init_program_path)
        language = extract_code_language(code)

        # 生成唯一ID并评估初始程序
        id = str(uuid.uuid4())
        # metrics = asyncio.run(direct_evaluate(config.evalutor_file_path, config.init_program_path, config))
        # initial_program = Program(
        #     id=id,
        #     code=code,
        #     language=language,
        #     parent_id=None,
        #     generation=0,
        #     timestamp=time.time(),
        #     metrics=metrics,
        #     iteration_found=0,
        # )

        # 初始化岛屿相关数据结构
        num_islands = config.island.num_islands
        islands_id = [str(i) for i in range(num_islands)] # 岛屿的id 全局唯一 不会被更新 安全
        best_program_each_island = {island_id:id for island_id in islands_id} # 每个岛屿上最好的程序id 安全 e.g. {"island_id":["program_id1","program_id2"]}
        generation_count = {island_id:0 for island_id in islands_id} # 每一个岛屿当前的代数 安全 e.g. {"island_id":0}
        island_programs = {island_id:ThreadSafePrograms() for island_id in islands_id} # 各个岛屿上的程序id 安全 e.g. {"island_id":["program_id1","program_id2"]}
        
        all_programs = ThreadSafePrograms()
        # all_programs.add_program(id,initial_program)
        
        newest_programs = {island_id:id for island_id in islands_id} # 各个岛屿上最新的程序id 安全 e.g. {"island_id":"program_id"}
        status = {island_id:IslandStatus.INIT_STATE.value for island_id in islands_id} # 初始化每个岛屿的状态 安全 e.g. {"island_id":IslandStatus.INIT_STATE}
        
        archive = ThreadSafePrograms()
        # archive.add_program(id,initial_program)
        
        island_generation_count = {island_id:0 for island_id in islands_id} # 各个岛屿当前的代数 安全 e.g. {"island_id":0}
        # island_evolution_direction = config.island.evolution_direction # 岛屿的进化方向 安全 e.g. {"island_id":"evolution_direction"}
        generation_count_in_meeting = 0 # 交流会进行的次数
        time_of_meeting = config.island.time_of_meeting # 每当各个岛屿迭代了time_of_meeting次 就会进行一次交流会 安全
        
        current_program_id = {str(i):id for i in range(num_islands)} # 各个岛屿上当前的程序id(child id) 安全 e.g. {"island_id":"program_id"}
        current_program_code = {str(i):code for i in range(num_islands)} # 各个岛屿上当前的程序代码 安全 e.g. {"island_id":"program_code"}
        sample_program_id = {island_id:"" for island_id in islands_id} # 各个岛屿上采样的父代程序id 安全 e.g. {"island_id":"program_id"}
        sample_inspirations = {island_id:[] for island_id in islands_id} # 各个岛屿上采样的程序的灵感程序id 安全 e.g. {"island_id":["program_id1","program_id2"]}
        artifacts = {island_id:{} for island_id in islands_id} # 从采样的父代程序得到的工件
        prompt = {island_id:"" for island_id in islands_id} # 各个岛屿上构建的提示词 安全 e.g. {"island_id":"prompt"}
        sample_top_programs = {island_id:[] for island_id in islands_id} # 各个岛屿上采样的最好的程序id 安全 e.g. {"island_id":["program_id1","program_id2"]}
        feature_map = {island_id:{} for island_id in islands_id} # 各个岛屿上的特征 安全 e.g. {"island_id":{"feature_name":feature_value}}
        
        llm_generate_success = {island_id:False for island_id in islands_id} # 各个岛屿上LLM生成是否成功 安全 e.g. {"island_id":False}
        llm_response = {island_id:"" for island_id in islands_id} # 各个岛屿上LLM的响应 安全 e.g. {"island_id":{"key":response}}
        llm_message_diff = {island_id:"" for island_id in islands_id} # 各个岛屿上LLM的diff响应 安全 e.g. {"island_id":{"key":response}}
        llm_message_rewrite = {island_id:"" for island_id in islands_id} # 各个岛屿上LLM的rewrite响应 安全 e.g. {"island_id":{"key":response}}
        llm_message_suggestion = {island_id:"" for island_id in islands_id} # 各个岛屿上LLM的suggestion响应 安全 e.g. {"island_id":{"key":response}}
        llm_change_summary = {island_id:"" for island_id in islands_id} # 各个岛屿上LLM的change_summary 安全 e.g. {"island_id":{"key":response}}
        
        return {
            "init_program":id,
            "best_program":code,
            "best_program_id":id,
            "best_program_each_island":best_program_each_island,
            "best_metrics":{},
            "generation_count":generation_count,
            "num_islands":num_islands,
            "archive":archive,
            "island_programs":island_programs,
            "all_programs":all_programs,
            "evaluation_program":config.evalutor_file_path,
            "newest_programs":newest_programs,
            "language":language,
            "file_extension":file_extension,
            "status":status,
            "island_generation_count":island_generation_count,
            "islands_id":islands_id,
            # island_evolution_direction=island_evolution_direction,
            "generation_count_in_meeting":generation_count_in_meeting,
            "time_of_meeting":time_of_meeting,
            "current_program_id":current_program_id,
            "current_program_code":current_program_code,
            "sample_program_id":sample_program_id,
            "sample_inspirations":sample_inspirations,
            "artifacts":artifacts,
            "prompt":prompt,
            "sample_top_programs":sample_top_programs,
            "feature_map":feature_map,
            "llm_response":llm_response, # 此属性未添加
            "llm_generate_success":llm_generate_success,
            "llm_message_diff":llm_message_diff,
            "llm_message_rewrite":llm_message_rewrite,
            "llm_message_suggestion":llm_message_suggestion,
            "llm_change_summary":llm_change_summary
            } 
    def __call__(self,state:GraphState):
        return self.execute(state)
            

def get_top_programs(state:GraphState, n: int = 10,metric:Optional[str] = None) -> List[Program]:
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
def get_artifacts(state:GraphState,program_id: str) -> Dict[str, Union[str, bytes]]:

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

class node_sample_parent_inspiration(SyncNode): 
    '''
    采样父代程序与灵感程序
    '''
    def __init__(self,config:Config,island_id:str,n:int=5,metric:Optional[str] = None):
        self.config = config 
        self.island_id = island_id 
        self.n = n 
        self.metric = metric 
        
    def execute(self,state:GraphState):
        try:
            parent = self._sample_parent(state)
            if parent is None:
                logger.error(f"采样父代程序失败: 返回None")
                return None,[]
            parent_id = parent.id 
        except Exception as e:
            logger.error(f"采样父代程序失败: {e}")
            return None,[]
        
        inspirations = self._sample_inspirations(state,parent_id)
        return parent_id,inspirations
    
    def __call__(self, state: GraphState, config: Optional[Any] = None) -> Dict[str, Any]:
        """
        LangGraph节点调用接口 - 线程安全版本
        只返回需要更新的字段，不直接修改state对象
        """
        # 执行采样逻辑
        parent_id,inspirations = self.execute(state)
        
        # 只返回需要更新的字段，让LangGraph的reducer处理并发
        print(f"岛屿{self.island_id}的采样父代程序与灵感程序完成")
        return {
            "sample_program_id": (self.island_id,parent_id),
            "sample_inspirations": (self.island_id,inspirations),
            "status": (self.island_id,IslandStatus.SAMPLE),
        }
    
    def _sample_inspirations(self,state:GraphState,parent_id:str):
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
            state.best_program_id is not None
            and state.best_program_id != parent_id
            and state.best_program_id in state.all_programs.get_program_ids()
        ):
            
            inspirations.append(state.best_program_id)
            logger.debug(f"Including best program {state.best_program_id} in inspirations")
        
        # 添加顶级程序作为灵感
        top_n = max(1, int(self.n * self.config.island.elite_selection_ratio))
        top_programs = get_top_programs(state,n=top_n,metric=self.metric)
        for program in top_programs:
            if program.id not in [p.id for p in inspirations] and program.id != parent_id:
                inspirations.append(program)
                
        
        # 添加多样性程序
        if len(state.all_programs) > self.n and len(inspirations) < self.n:
            # 计算要添加的多样性程序数量（最多到剩余位置）
            remaining_slots = self.n - len(inspirations)

            # 从不同的特征格子采样以获得多样性

            feature_coords = _calculate_feature_coords(self.config,state,state.all_programs.get_program(parent_id))

            # 从附近的特征格子获取程序
            nearby_programs = []
            for _ in range(remaining_slots):
                # 扰动坐标
                perturbed_coords = [
                    max(0, min(self.config.island.feature_bins - 1, c + random.randint(-1, 1)))
                    for c in feature_coords
                ]

                # 尝试从这个格子获取程序
                cell_key = _feature_coords_to_key(perturbed_coords)
                if cell_key in state.feature_map:
                    program_id = state.feature_map[cell_key]
                    # 在添加前检查程序是否仍然存在
                    if (
                        program_id != parent_id
                        and program_id not in [p.id for p in inspirations]
                        and program_id in state.all_programs
                    ):
                        nearby_programs.append(state.all_programs.get_program(program_id))
                    elif program_id not in state.all_programs:
                        # 清理特征网格中的过时引用
                        logger.debug(f"Removing stale program {program_id} from feature_map")
                        del state.feature_map[cell_key]

            # 如果需要更多，添加随机程序
            if len(inspirations) + len(nearby_programs) < self.n:
                remaining = self.n - len(inspirations) - len(nearby_programs)
                all_ids = set(state.all_programs.get_program_ids())
                excluded_ids = (
                    {parent_id}
                    .union(p.id for p in inspirations)
                    .union(p.id for p in nearby_programs)
                )
                available_ids = list(all_ids - excluded_ids)

                if available_ids:
                    random_ids = random.sample(available_ids, min(remaining, len(available_ids)))
                    random_programs = [state.all_programs.get_program(pid) for pid in random_ids]
                    nearby_programs.extend(random_programs)

            inspirations.extend(nearby_programs)

        return inspirations[:self.n]
    def _sample_parent(self,state:GraphState) -> Optional[Program | None | Any]:
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
            # 探索：从当前岛屿采样（多样性采样）
            return self._sample_exploration_parent(state)
        elif rand_val < self.config.island.exploration_ratio + self.config.island.exploitation_ratio:
            # 利用：从归档采样（精英程序）
            return self._sample_exploitation_parent(state)
        else:
            # 随机：从任何程序采样（剩余概率）
            return self._sample_random_parent(state)
    def _sample_exploration_parent(self, state: GraphState) -> Optional[Program | None | Any]:
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
        current_island_programs = state.island_programs[self.island_id].get_program_ids()
        
        # 岛屿完整性验证
        # 正常情况下每个岛屿都应该包含至少一个初始程序
        if not current_island_programs:
            raise ValueError(
                f"岛屿 {self.island_id} 未正确初始化，程序列表为空。"
                f"请确保初始程序已正确设置到所有岛屿中。"
            )
        
        # 探索性随机采样
        # 使用random.choice确保每个程序被选中的概率相等
        parent_id = random.choice(list(current_island_programs))
        
        # 从全局程序字典中获取完整的Program对象
        # 这里假设程序ID的一致性已经在其他地方得到保证
        
        return state.all_programs.get(parent_id)

    def _sample_exploitation_parent(self, state: GraphState) -> Optional[Program | None | Any]:
        """ 
        利用性采样父代程序（从精英归档采样策略）
        
        该方法实现了岛屿模型中的利用性采样策略，用于从精英归档中
        选择一个程序作为下一轮进化的父代程序。这种采样方式有助于
        利用已有的优秀程序，加速收敛到全局最优解。
        
        算法流程：
        """
        
        archive_programs_ids = state.archive.get_program_ids() # 精英归档的id集合 archive在初始化时候一定会被初始化 所以一定有值 
        
        # 尽可能获得当前岛屿内部的精英归档
        archive_in_current_island = [pid for pid in state.archive.get_program_ids() if pid in state.island_programs[self.island_id]]
        #优先从自己的岛屿上采样 如果自己的岛屿上没有精英归档类别的程序 就随机采样 
        
        if len(archive_in_current_island) > 0: #如果自己的岛屿上有精英归档类别的程序 则从自己的岛屿上采样
            return state.all_programs.get_program(random.choice(archive_in_current_island))
        else: #如果自己的岛屿上没有精英归档类别的程序 则从所有岛屿的精英归档中采样
            if len(archive_programs_ids) > 0:
                return state.all_programs.get_program(random.choice(archive_programs_ids))
            else:
                return self._sample_random_parent(state)
            
    def _sample_random_parent(self,state:GraphState) -> Optional[Program | None | Any]:
        """
        完全随机采样父代程序
        
        Returns:
            Program: 选中的父代程序
        """
        if not state.all_programs:
            raise ValueError("No programs available for sampling")

        # 从自己岛屿上的所有程序中随机采样
        current_island_programs = state.island_programs[self.island_id].get_program_ids()
        if not current_island_programs:
            raise ValueError(f"当前岛屿 {self.island_id} 的程序列表为空")
        program_id = random.choice(list(current_island_programs))
        return state.all_programs.get_program(program_id)
   
class node_build_prompt(SyncNode):
    '''
    这个节点根据state中的信息来构建prompt
    
    '''
    def __init__(self,config:Config,island_id:str,metric:Optional[str] = None):
        self.config = config 
        self.island_id = island_id 
        self.metric = metric 
        self.prompt_sampler = PromptSampler_langchain(config=config.prompt)
    def execute(self,state:GraphState):
        return self._build_prompt(state)
    def __call__(self, state: GraphState, config: Optional[Any] = None) -> Dict[str, Any]:
        prompt = self.execute(state)
        print(f"岛屿{self.island_id}的构建prompt完成")
        return {
            "prompt": (self.island_id,prompt),
            "status": (self.island_id,IslandStatus.BUILD_PROMPT)
        }
    def _build_prompt(self,state:GraphState)->str:
        
        # 若当前program为空 则使用父代程序
        current_program_id = state.current_program_id[self.island_id] if state.current_program_id[self.island_id] is not None else state.sample_program_id[self.island_id]
        current_program = state.all_programs.get(current_program_id)
        if current_program is None:
            logger.error(f"当前程序为空: {current_program_id}")
            return ""
        
        parent_program = state.all_programs.get(state.sample_program_id[self.island_id])
        if parent_program is None:
            logger.error(f"父代程序为空: {state.sample_program_id[self.island_id]}")
            return ""
        parent_code = parent_program.code
        parent_metrics = parent_program.metrics
        
        previous_programs = []
        #当前的父代程序
        parent_program = state.all_programs.get(state.sample_program_id[self.island_id])
        for _ in range(3):
            #如果父代程序有父代程序 则将父代程序的父代程序加入previous_programs 否则结束循环
            if isinstance(parent_program,Program) and parent_program.parent_id:
                previous_programs.append(state.all_programs.get(parent_program.parent_id))
                parent_program = state.all_programs.get(parent_program.parent_id)
            else:
                break
        if previous_programs != []:
            previous_programs = [state.all_programs.get(pid).to_dict() for pid in previous_programs if pid is not None]
        else:
            previous_programs = []
            
            
        
        inspirations = state.sample_inspirations[self.island_id]
        inspirations_programs = [state.all_programs.get(pid).to_dict() for pid in inspirations]
        
        top_programs = [i.to_dict() for i in get_top_programs(state,n=5,metric=self.metric)]
        # previous_programs = 
        
        # import pdb;pdb.set_trace()
        
        return self.prompt_sampler.build_prompt(
            current_program = current_program.code,
            parent_program = parent_code,
            program_metrics = parent_metrics,  # 程序指标字典
            previous_protgrams = previous_programs,  # 之前的程序尝试列表 即这个岛屿上的父代程序的集合 取三代 （即当前父代的前二代）
            top_programs = top_programs,       # 顶级程序列表（按性能排序）
            inspirations = inspirations_programs,       # 灵感程序列表
            language = "python",            # 编程语言
            evolution_round = state.generation_count[self.island_id],            # 演化轮次
            diff_based_evolution = self.config.diff_based_evolution,   # 是否使用基于差异的演化
            program_artifacts = get_artifacts(state,current_program_id),  # 程序工件
        )
        
class node_llm_generate(SyncNode):
    def __init__(self,config:Config,island_id:str):
        self.config = config 
        self.island_id = island_id 
        self.llm = LLMs(config=self.config)
        self.diff_based_evolution = self.config.diff_based_evolution
        if self.diff_based_evolution: #如果是基于差异的演化 那么使用
            self.structure = ResponseFormatter_template_diff
            self.key = ["suggestion","diff_code"]
        else:
            self.structure = ResponseFormatter_template_rewrite
            self.key = ["suggestion","rewrite_code"]
             
    def _llm_generate(self,state:GraphState):
        return self.llm.invoke(state.prompt[self.island_id],self.structure,self.key)
    async def execute(self,state:GraphState):
        llm_response = await self._llm_generate(state)
        return llm_response
    def __call__(self,state:GraphState):
        print(f"岛屿{self.island_id}的LLM生成开始")
        
        # 如果llm_response为None 则为超时或者其他原因
        
        llm_response = asyncio.run(self.execute(state))
        
        print(f"岛屿{self.island_id}的LLM生成完成")
        if llm_response is None:
            return None
        
        if self.diff_based_evolution:
            diff_code = llm_response["diff_code"]
            suggestion = llm_response["suggestion"]
            return {
                "llm_message_diff":(self.island_id,diff_code),
                "llm_message_suggestion":(self.island_id,suggestion),
                "status": (self.island_id,IslandStatus.LLM_GENERATE),
                "llm_generate_success":(self.island_id,True)
            }
        else:
            rewrite_code = llm_response["rewrite_code"]
            suggestion = llm_response["suggestion"]
            return {
                "llm_message_rewrite":(self.island_id,rewrite_code),
                "llm_message_suggestion":(self.island_id,suggestion),
                "status": (self.island_id,IslandStatus.LLM_GENERATE),
                "llm_generate_success":(self.island_id,True)
            }
        
class node_generate_child(SyncNode):
    '''
    根据llm的输出 生成子代程序 这个节点不评估 只生成子代程序 生成后更新current_program
    '''
    def __init__(self,config:Config,island_id:str):
        self.config = config 
        self.island_id = island_id 
        self.llm = LLMs(config=self.config)
        self.diff_based_evolution = self.config.diff_based_evolution
        
    def diff_evolution(self,parent_code:str,llm_message_diff:str,state:GraphState) -> Optional[Tuple[str,str] | Tuple[str,None]]:
        diff_blocks = extract_diffs(llm_message_diff)
        if not diff_blocks:
            logger.warning(f"岛屿{self.island_id}第{state.generation_count[self.island_id]}轮LLM的输出没有diff_blocks")
            return parent_code,None
        
        childe_code = apply_diff(parent_code,llm_message_diff)
        change_summary = format_diff_summary(diff_blocks)
        
        return childe_code,change_summary
    
    def rewrite_evolution(self,parent_code:str,llm_message_rewrite:str,state:GraphState) -> Optional[Tuple[str,str] | Tuple[str,None]]:
        new_code = parse_full_rewrite(llm_message_rewrite,state.language)
        if not new_code:
            logger.warning(f"岛屿{self.island_id}第{state.generation_count[self.island_id]}轮LLM的输出没有new_code")
            return parent_code,None
        change_summary = "full rewrite"
        return new_code,change_summary
    
    def execute(self,state:GraphState):
        parent_code = state.all_programs.get(state.current_program_id[self.island_id]).code #先获得父代的代码
        if self.diff_based_evolution:
            child_code,change_summary = self.diff_evolution(parent_code,state.llm_message_diff[self.island_id],state)
        else:
            child_code,change_summary = self.rewrite_evolution(parent_code,state.llm_message_rewrite[self.island_id],state)
        return child_code,change_summary
    def __call__(self,state:GraphState):
        child_code,change_summary = self.execute(state)
        print(f"岛屿{self.island_id}的生成子代程序完成")
        return {
            "current_program":(self.island_id,child_code),
            "llm_change_summary":(self.island_id,change_summary),
            "status": (self.island_id,IslandStatus.GENERATE_CHILD)
        }
class node_add_program(SyncNode):
    '''
    将子代程序添加到程序库中 并更新state 
    '''
    def __init__(self,config:Config,island_id:str):
        self.config = config 
        self.island_id = island_id 
        
    def execute(self,state:GraphState):
        return self._add_program(state)
    
    def __call__(self,state:GraphState):
        return self.execute(state)
    
    def _add_program(self,state:GraphState):
        #首先获得子代程序的code
        child_code = state.current_program_code[self.island_id]
        
        #生成其id
        id = str(uuid.uuid4())
        #评估子代程序
        metrics = asyncio.run(direct_evaluate(self.config.evalutor_file_path, self.config.init_program_path, self.config))
        language = extract_code_language(child_code)
        # 创建初始程序对象
        child_program = Program(
            id=id,
            code=child_code,
            language=language,
            parent_id=state.sample_program_id[self.island_id],
            generation=state.generation_count[self.island_id],
            timestamp=time.time(),
            metrics=metrics,
            iteration_found=state.generation_count[self.island_id],
        )
        
        # 判定是否要更新其他内容
        
        
        
        
        
        






if __name__ == "__main__":
    from langgraph.graph import StateGraph,START,END
    import time
    
    config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
    # print(graph_state)
    builder = StateGraph(GraphState)
    
    # print(f"岛屿ID列表: {graph_state.islands_id}")
    
    def wait_node(state:GraphState):
        print(f"等待节点执行时间: {time.time()}")
        return state
    
    
    init_node = node_init_status(config=config)
    init_node_evaluate = node_evaluate(config=config,island_id=None)
    builder.add_node("init_node",init_node)
    builder.add_node("init_node_evaluate",init_node_evaluate)
    builder.add_edge(START,"init_node")
    builder.add_edge("init_node","init_node_evaluate")
    builder.add_edge("init_node_evaluate",END)
    
    graph=builder.compile()
    result = graph.invoke(GraphState())
    print(result)
    import pdb;pdb.set_trace()
    

    
    
    
    

    # # # # 创建各个岛屿的节点实例
    # node_sample_parent_1 = node_sample_parent_inspiration(config=config,island_id=graph_state.islands_id[0])
    # node_sample_parent_2 = node_sample_parent_inspiration(config=config,island_id=graph_state.islands_id[1])
    # node_sample_parent_3 = node_sample_parent_inspiration(config=config,island_id=graph_state.islands_id[2])
    # node_sample_parent_4 = node_sample_parent_inspiration(config=config,island_id=graph_state.islands_id[3])
    
    # node_build_prompt_1 = node_build_prompt(config=config,island_id=graph_state.islands_id[0],metric=None)
    # node_build_prompt_2 = node_build_prompt(config=config,island_id=graph_state.islands_id[1],metric=None)
    # node_build_prompt_3 = node_build_prompt(config=config,island_id=graph_state.islands_id[2],metric=None)
    # node_build_prompt_4 = node_build_prompt(config=config,island_id=graph_state.islands_id[3],metric=None)
    
    
    # builder.add_node("node_sample_parent_1",node_sample_parent_1)
    # builder.add_node("node_sample_parent_2",node_sample_parent_2)
    # builder.add_node("node_sample_parent_3",node_sample_parent_3)
    # builder.add_node("node_sample_parent_4",node_sample_parent_4)
    
    # # builder.add_node("node_build_prompt_1",node_build_prompt_1)
    # # builder.add_node("node_build_prompt_2",node_build_prompt_2)
    # # builder.add_node("node_build_prompt_3",node_build_prompt_3)
    # # builder.add_node("node_build_prompt_4",node_build_prompt_4)
    
    # builder.add_node("wait_node",wait_node,defer = True)
    
    # # # # 添加边连接 - 这里展示了并行执行的关键
    # # # # 从START同时启动4个岛屿的采样父代节点 - 这些节点将并行执行
    # # # print("设置并行执行的边连接:")
    # # # print("1. 从START同时连接到4个采样父代节点 - 实现并行启动")
    # builder.add_edge(START,"init_node")
    # builder.add_edge("init_node","node_sample_parent_1")
    # builder.add_edge("init_node","node_sample_parent_2") 
    # builder.add_edge("init_node","node_sample_parent_3")
    # builder.add_edge("init_node","node_sample_parent_4")
    # builder.add_edge("node_sample_parent_1","wait_node")
    # builder.add_edge("node_sample_parent_2","wait_node")
    # builder.add_edge("node_sample_parent_3","wait_node")
    # builder.add_edge("node_sample_parent_4","wait_node")
    # # builder.add_edge("node_sample_parent_1","node_build_prompt_1")
    # # builder.add_edge("node_sample_parent_2","node_build_prompt_2")
    # # builder.add_edge("node_sample_parent_3","node_build_prompt_3")
    # # builder.add_edge("node_sample_parent_4","node_build_prompt_4")
    
    # # builder.add_edge("node_build_prompt_1","wait_node")
    # # builder.add_edge("node_build_prompt_2","wait_node")
    # # builder.add_edge("node_build_prompt_3","wait_node")
    # # builder.add_edge("node_build_prompt_4","wait_node")
    
    # builder.add_edge("wait_node",END)
    
    # graph = builder.compile()
    # result = graph.invoke(graph_state)
    
    # print(result)