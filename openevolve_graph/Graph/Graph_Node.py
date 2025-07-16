from Graph_Node_ABC import *
from openevolve_graph.Graph.Graph_Node_ABC import NodeType
from openevolve_graph.Graph.Graph_state import GraphState 
from openevolve_graph.Config import Config
from openevolve_graph.utils.utils import safe_numeric_average 
from openevolve_graph.utils.utils import _calculate_feature_coords,_feature_coords_to_key


class node_sample_parent(SyncNode): 
    '''
    采样父代程序
    '''
    def __init__(self,config:Config,island_id:str):
        self.config = config 
        self.island_id = island_id 
        
    def execute(self,state:GraphState):
        return self._sample_parent(state) 
    
    def get_node_type(self) -> NodeType:
        return super().get_node_type()
    def validate_input(self, state: GraphState) -> bool:
        return super().validate_input(state)
    def handle_error(self, error: Exception, state: GraphState) -> NodeResult:
        return super().handle_error(error, state)
    
    def __call__(self, state: GraphState, config: Optional[Any] = None) -> Dict[str, Any]:
        """
        LangGraph节点调用接口 - 线程安全版本
        只返回需要更新的字段，不直接修改state对象
        """
        # 执行采样逻辑
        program_id: str = self.execute(state).id
        
        # 只返回需要更新的字段，让LangGraph的reducer处理并发
        return {
            "sample_program_id": {self.island_id: program_id}
        }
    
    def get_node_info(self) -> Dict[str, Any]:
        return super().get_node_info()
    
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
        return state.all_programs[parent_id]

    def _sample_exploitation_parent(self, state: GraphState) -> Program:
        """ 
        利用性采样父代程序（从精英归档采样策略）
        
        该方法实现了岛屿模型中的利用性采样策略，用于从精英归档中
        选择一个程序作为下一轮进化的父代程序。这种采样方式有助于
        利用已有的优秀程序，加速收敛到全局最优解。
        
        算法流程：
        """
        
        archive_programs = state.archive # 精英归档的id集合 archive在初始化时候一定会被初始化 所以一定有值 
        archive_in_current_island = [pid for pid in archive_programs if pid in state.island_programs[self.island_id]]
        
        #优先从自己的岛屿上采样 如果自己的岛屿上没有精英归档类别的程序 就随机采样 
        if len(archive_in_current_island) > 0:
            return state.all_programs[random.choice(archive_in_current_island)]
        else:
            return state.all_programs[random.choice(archive_programs)]
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
        return state.all_programs[program_id]

class node_sample_inspiration(SyncNode):
    '''
    采样灵感程序
    '''
    def __init__(self,config:Config,island_id:str,n:int,metric:Optional[str] = None):
        self.config = config 
        self.island_id = island_id 
        self.n = n 
        self.metric = metric 
    def execute(self,state:GraphState):
        return self._sample_parent(state) 
    
    def get_node_type(self) -> NodeType:
        return super().get_node_type()
    def validate_input(self, state: GraphState) -> bool:
        return super().validate_input(state)
    def handle_error(self, error: Exception, state: GraphState) -> NodeResult:
        return super().handle_error(error, state)
    
    def __call__(self, state: GraphState, config: Optional[Any] = None) -> Dict[str, Any]:
        """
        LangGraph节点调用接口 - 线程安全版本
        只返回需要更新的字段，不直接修改state对象
        """
        # 执行采样逻辑
        program_id: str = self.execute(state).id
        
        # 只返回需要更新的字段，让LangGraph的reducer处理并发
        return {
            "sample_program_id": {self.island_id: program_id}
        }
    
    def get_node_info(self) -> Dict[str, Any]:
        return super().get_node_info()
    
    def get_top_programs(self, state:GraphState, n: int = 10) -> List[Program]:
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

        if self.metric:
            # 按指定指标排序
            sorted_programs = sorted(
                [p for p in state.all_programs.values() if self.metric in p.metrics],
                key=lambda p: p.metrics[self.metric],
                reverse=True,
            )
        else:
            # 按所有数值指标的平均值排序
            sorted_programs = sorted(
                state.all_programs.values(),
                key=lambda p: safe_numeric_average(p.metrics),
                reverse=True,
            )

        return sorted_programs[:n]
    def _sample_inspirations(self,state:GraphState):
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
        parent_id = state.sample_program_id[self.island_id]
        
        #若最优程序存在 且与父代不同 且在所有的程序中 则加入灵感程序
        if (
            state.best_program_id is not None
            and state.best_program_id != parent_id
            and state.best_program_id in state.all_programs
        ):
            
            inspirations.append(state.best_program_id)
            logger.debug(f"Including best program {state.best_program_id} in inspirations")
        
        # 添加顶级程序作为灵感
        top_n = max(1, int(self.n * self.config.island.elite_selection_ratio))
        top_programs = self.get_top_programs(state,n=top_n)
        for program in top_programs:
            if program.id not in [p.id for p in inspirations] and program.id != parent_id:
                inspirations.append(program)
                
        
        # 添加多样性程序
        if len(state.all_programs) > self.n and len(inspirations) < self.n:
            # 计算要添加的多样性程序数量（最多到剩余位置）
            remaining_slots = self.n - len(inspirations)

            # 从不同的特征格子采样以获得多样性
            feature_coords = _calculate_feature_coords(self.config,state,state.all_programs[parent_id])

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
                        program_id != parent.id
                        and program_id not in [p.id for p in inspirations]
                        and program_id in self.programs
                    ):
                        nearby_programs.append(self.programs[program_id])
                    elif program_id not in self.programs:
                        # 清理特征网格中的过时引用
                        logger.debug(f"Removing stale program {program_id} from feature_map")
                        del self.feature_map[cell_key]

            # 如果需要更多，添加随机程序
            if len(inspirations) + len(nearby_programs) < n:
                remaining = n - len(inspirations) - len(nearby_programs)
                all_ids = set(self.programs.keys())
                excluded_ids = (
                    {parent.id}
                    .union(p.id for p in inspirations)
                    .union(p.id for p in nearby_programs)
                )
                available_ids = list(all_ids - excluded_ids)

                if available_ids:
                    random_ids = random.sample(available_ids, min(remaining, len(available_ids)))
                    random_programs = [self.programs[pid] for pid in random_ids]
                    nearby_programs.extend(random_programs)

            inspirations.extend(nearby_programs)

        return inspirations[:n]


        
        
    
if __name__ == "__main__":
    from openevolve_graph.Graph.Graph_state import init_graph_state
    from langgraph.graph import StateGraph,START,END
    config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
    graph_state = init_graph_state(config)
    print(graph_state.islands_id)
    
    def wait_node(state:GraphState):
        return state
    # state = GraphState()
    node_sample_parent_1 = node_sample_parent(config=config,island_id=graph_state.islands_id[0])
    node_sample_parent_2 = node_sample_parent(config=config,island_id=graph_state.islands_id[1])
    node_sample_parent_3 = node_sample_parent(config=config,island_id=graph_state.islands_id[2])
    node_sample_parent_4 = node_sample_parent(config=config,island_id=graph_state.islands_id[3])
    
    
    builder = StateGraph(GraphState)
    builder.add_node("node_sample_parent_1",node_sample_parent_1)
    builder.add_node("node_sample_parent_2",node_sample_parent_2)
    builder.add_node("node_sample_parent_3",node_sample_parent_3)
    builder.add_node("node_sample_parent_4",node_sample_parent_4)
    builder.add_node("wait_node",wait_node,defer=True)
    builder.add_edge(START,"node_sample_parent_1")
    builder.add_edge(START,"node_sample_parent_2")
    builder.add_edge(START,"node_sample_parent_3")
    builder.add_edge(START,"node_sample_parent_4")
    builder.add_edge("node_sample_parent_1","wait_node")
    builder.add_edge("node_sample_parent_2","wait_node")
    builder.add_edge("node_sample_parent_3","wait_node")
    builder.add_edge("node_sample_parent_4","wait_node")
    builder.add_edge("wait_node",END)
    
    graph = builder.compile()
    print(graph.invoke(graph_state))
    
    #successful
