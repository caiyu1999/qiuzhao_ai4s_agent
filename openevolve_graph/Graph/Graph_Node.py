import os 
import json 
from typing import Dict
import uuid

from pydantic import BaseModel
from Graph_Node_ABC import *
# from openevolve_graph.Graph.Graph_Node_ABC import NodeType
from openevolve_graph.Graph.Graph_Node_ABC import NodeResult
from openevolve_graph.Graph.Graph_state import GraphState, IslandStatus 
from openevolve_graph.Config import Config
from openevolve_graph.utils.utils import safe_numeric_average 
from openevolve_graph.utils.utils import _calculate_feature_coords,_feature_coords_to_key
from openevolve_graph.Prompt.sampler import PromptSampler_langchain
from openevolve_graph.utils.utils import load_initial_program, extract_code_language
from openevolve_graph.utils.thread_safe_programs import ThreadSafePrograms
from evaluator import direct_evaluate
from torch import Value
from typing_extensions import TypedDict
import time
logger = logging.getLogger(__name__)





class node_defer(SyncNode):
    def __init__(self,config:Config):
        self.config = config
        
    def execute(self,state:GraphState):
        return state
    
    def __call__(self,state:GraphState):
        return state



class node_init_status(SyncNode):
    '''
    初始化图的状态
    确保所有GraphState属性都被正确初始化
    '''
    
    def __init__(self,config:Config):
        self.config = config
        self.num_islands = config.island.num_islands
        
    def execute(self, state: BaseModel) -> NodeResult | Dict[str, Any]:
        return super().execute(state)
        
    def __call__(self,state:GraphState):
    
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
        metrics = asyncio.run(direct_evaluate(config.evalutor_file_path, config.init_program_path, config))

        # 创建初始程序对象
        initial_program = Program(
            id=id,
            code=code,
            language=language,
            parent_id=None,
            generation=0,
            timestamp=time.time(),
            metrics=metrics,
            iteration_found=0,
        )

        # 初始化岛屿相关数据结构
        num_islands = config.island.num_islands
        islands_id = [str(i) for i in range(num_islands)] # 岛屿的id 全局唯一 不会被更新 安全
        best_program_each_island = {island_id:id for island_id in islands_id} # 每个岛屿上最好的程序id 安全 e.g. {"island_id":["program_id1","program_id2"]}
        generation_count = {island_id:0 for island_id in islands_id} # 每一个岛屿当前的代数 安全 e.g. {"island_id":0}
        island_programs = {island_id:[id] for island_id in islands_id} # 各个岛屿上的程序id 安全 e.g. {"island_id":["program_id1","program_id2"]}
        all_programs = ThreadSafePrograms()
        all_programs.add_program(id,initial_program)
        newest_programs = {island_id:id for island_id in islands_id} # 各个岛屿上最新的程序id 安全 e.g. {"island_id":"program_id"}
        status = {island_id:IslandStatus.INIT_STATE.value for island_id in islands_id} # 初始化每个岛屿的状态 安全 e.g. {"island_id":IslandStatus.INIT_STATE}
        archive = ThreadSafePrograms()
        archive.add_program(id,initial_program)
        island_generation_count = {island_id:0 for island_id in islands_id} # 各个岛屿当前的代数 安全 e.g. {"island_id":0}
        # island_evolution_direction = config.island.evolution_direction # 岛屿的进化方向 安全 e.g. {"island_id":"evolution_direction"}
        generation_count_in_meeting = 0 # 交流会进行的次数
        time_of_meeting = config.island.time_of_meeting # 每当各个岛屿迭代了time_of_meeting次 就会进行一次交流会 安全
        
        current_program_id = {island_id:id for island_id in islands_id} # 各个岛屿上当前的程序id(child id) 安全 e.g. {"island_id":"program_id"}
        sample_program_id = {island_id:"" for island_id in islands_id} # 各个岛屿上采样的父代程序id 安全 e.g. {"island_id":"program_id"}
        sample_inspirations = {island_id:[] for island_id in islands_id} # 各个岛屿上采样的程序的灵感程序id 安全 e.g. {"island_id":["program_id1","program_id2"]}
        artifacts = {} # 从采样的父代程序得到的工件
        prompt = {island_id:"" for island_id in islands_id} # 各个岛屿上构建的提示词 安全 e.g. {"island_id":"prompt"}
        sample_top_programs = {island_id:[] for island_id in islands_id} # 各个岛屿上采样的最好的程序id 安全 e.g. {"island_id":["program_id1","program_id2"]}
        feature_map = {island_id:{} for island_id in islands_id} # 各个岛屿上的特征 安全 e.g. {"island_id":{"feature_name":feature_value}}
        
        # 确保所有GraphState属性都被正确初始化
        # print("init_program",id,'\n',
        #     "best_program",code,'\n',
        #     "best_program_id",id,'\n',
        #     "best_program_each_island",best_program_each_island,'\n',
        #     "best_metrics",metrics,'\n',
        #     "generation_count",generation_count,'\n',
        #     "num_islands",num_islands,'\n',
        #     "archive",archive,'\n',
        #     "island_programs",island_programs,'\n',
        #     "all_programs",all_programs,'\n',
        #     "evaluation_program",config.evalutor_file_path,'\n',
        #     "newest_programs",newest_programs,'\n',
        #     "language",language,'\n',
        #     "file_extension",file_extension,'\n',
        #     "status",status,'\n',
        #     "island_generation_count",island_generation_count,'\n',
        #     "islands_id",islands_id,'\n',
        #     # island_evolution_direction=island_evolution_direction,
        #     "generation_count_in_meeting",generation_count_in_meeting,'\n',
        #     "time_of_meeting",time_of_meeting,'\n',
        #     "current_program_id",current_program_id,'\n',
        #     "sample_program_id",sample_program_id,'\n',
        #     "sample_inspirations",sample_inspirations,'\n',
        #     "artifacts",artifacts,'\n',
        #     "prompt",prompt,'\n',
        #     "sample_top_programs",sample_top_programs,'\n',
        #     "feature_map",feature_map,'\n')
        
        return {
            "init_program":id,
            "best_program":code,
            "best_program_id":id,
            "best_program_each_island":best_program_each_island,
            "best_metrics":metrics,
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
            "sample_program_id":sample_program_id,
            "sample_inspirations":sample_inspirations,
            "artifacts":artifacts,
            "prompt":prompt,
            "sample_top_programs":sample_top_programs,
            "feature_map":feature_map
            }
            

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
        parent_id = self._sample_parent(state).id 
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
            and state.best_program_id in state.all_programs.get_all_ids()
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
    def _sample_parent(self,state:GraphState) -> Program:
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
    def _sample_exploration_parent(self, state: GraphState) -> Program:
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
        current_island_programs = state.island_programs[self.island_id]
        
        # 岛屿完整性验证
        # 正常情况下每个岛屿都应该包含至少一个初始程序
        if not current_island_programs:
            raise ValueError(
                f"岛屿 {self.island_id} 未正确初始化，程序列表为空。"
                f"请确保初始程序已正确设置到所有岛屿中。"
            )
        
        # 探索性随机采样
        # 使用random.choice确保每个程序被选中的概率相等
        parent_id = random.choice(current_island_programs)
        
        # 从全局程序字典中获取完整的Program对象
        # 这里假设程序ID的一致性已经在其他地方得到保证
        
        return state.all_programs.get(parent_id)

    def _sample_exploitation_parent(self, state: GraphState) -> Program:
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
            
    def _sample_random_parent(self,state:GraphState) -> Program:
        """
        完全随机采样父代程序
        
        Returns:
            Program: 选中的父代程序
        """
        if not state.all_programs:
            raise ValueError("No programs available for sampling")

        # 从自己岛屿上的所有程序中随机采样
        current_island_programs = state.island_programs[self.island_id]
        if not current_island_programs:
            raise ValueError(f"当前岛屿 {self.island_id} 的程序列表为空")
        program_id = random.choice(current_island_programs)
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
        return {
            "prompt": (self.island_id,prompt),
            "status": (self.island_id,IslandStatus.BUILD_PROMPT)
        }
    def _build_prompt(self,state:GraphState)->str:
        
        # 若当前program为空 则使用父代程序
        current_program_id = state.current_program_id[self.island_id] if state.current_program_id[self.island_id] is not None else state.sample_program_id[self.island_id]
        current_program = state.all_programs.get(current_program_id).code
       
        parent_code = state.all_programs.get(state.sample_program_id[self.island_id]).code
        parent_metrics = state.all_programs.get(state.sample_program_id[self.island_id]).metrics
        
        previous_programs = []
        #当前的父代程序
        parent_program = state.all_programs.get(state.sample_program_id[self.island_id])
        for _ in range(3):
            #如果父代程序有父代程序 则将父代程序的父代程序加入previous_programs 否则结束循环
            if parent_program.parent_id:
                previous_programs.append(state.all_programs.get(parent_program.parent_id))
                parent_program = state.all_programs.get(parent_program.parent_id)
            else:
                break
        if previous_programs != []:
            previous_programs = [state.all_programs.get(pid).to_dict() for pid in previous_programs]
        else:
            previous_programs = []
            
            
        
        inspirations = state.sample_inspirations[self.island_id]
        inspirations_programs = [state.all_programs.get(pid).to_dict() for pid in inspirations]
        
        top_programs = [i.to_dict() for i in get_top_programs(state,n=5,metric=self.metric)]
        # previous_programs = 
        
        # import pdb;pdb.set_trace()
        
        return self.prompt_sampler.build_prompt(
            current_program = current_program,
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
    builder.add_node("init_node",init_node)
    builder.add_edge(START,"init_node")
    builder.add_edge("init_node",END)
    
    graph=builder.compile()
    result = graph.invoke(GraphState())
    
    for k,v in result.items():
        print(f"{k}: {v}")
    
    
    
    

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