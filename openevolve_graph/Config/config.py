from openevolve_graph.Config.graph_config import GraphConfig 
from openevolve_graph.Config.island_config import IslandConfig 
from openevolve_graph.Config.llm_config import ModelConfig,LLMConfig
from openevolve_graph.Config.prompt_config import PromptConfig
from openevolve_graph.Config.evaluator_config import EvaluatorConfig
from openevolve_graph.Config.controller_config import ControllerConfig
from openevolve_graph.Config.rag_config import RAGConfig, EmbeddingsConfig, LLMRagConfig


from dataclasses import dataclass,field 
from typing import Optional,Dict,Any,Union
from pathlib import Path
import yaml
import os


@dataclass
class Config:
    llm:LLMConfig = field(default_factory=LLMConfig)
    graph:GraphConfig = field(default_factory=GraphConfig)
    island:IslandConfig = field(default_factory=IslandConfig)
    prompt:PromptConfig = field(default_factory=PromptConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    # General settings
    
    max_iterations: int = 10000
    checkpoint_interval: int = 100
    diff_based_evolution: bool = True
    allow_full_rewrites: bool = False
    enable_artifacts: bool = False # 是否开启工件 如果不开启 不存储工件信息
    random_seed: Optional[int] = 42
    max_code_length: int = 20000
    init_program_path: str = ""
    evalutor_file_path: str = ""
    # 工件存储
    artifacts_base_path: Optional[str] = None  # Defaults to db_path/artifacts
    artifact_size_threshold: int = 32 * 1024  # 32KB threshold
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30
    
    archive_size: int = 20
    random_meeting:bool = True # 是否开启随机meeting 如果开启 则每次meeting后 下一次meeting的时间随机 
    meeting_interval:int = 2 # 每次meeting的间隔 如果开启随机meeting 则每次meeting后 下一次meeting的时间随机 
    meeting_interval_list: list = field(default_factory=lambda: [2,4,8,16]) # meeting的间隔列表  如果未到达max_iterations 则使用最后的值 
    graph_image_path: str = ""
   
    #graph running configuration 
    checkpoint: str = ""
    log_dir: str = ""
    log_level: str = "INFO"
    resume: bool = True 
    resume_iteration: int = 0
    
    # port for visualization
    port: int = 9980 
    
    
    use_rag:bool = False # 是否使用rag
    rag_doc_path:str = "" # rag文档路径
    vector_save_dir:str = ""

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file"""
        with open(path, "r",encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary"""
        # Handle nested configurations
        config = Config()

        # Update top-level fields
        for key, value in config_dict.items():
            if key not in ["llm", "prompt", "island", "evaluator","controller","graph","rag"] and hasattr(config, key):
                setattr(config, key, value)

        # Update nested configs
        if "llm" in config_dict:
            llm_dict = config_dict["llm"].copy()
            
            # Handle models field (current format)
            if "models" in llm_dict and isinstance(llm_dict["models"], list):
                # Check if it's a list of dictionaries (object format)
                if llm_dict["models"] and isinstance(llm_dict["models"][0], dict):
                    llm_dict["models"] = [ModelConfig(**m) for m in llm_dict["models"]]
                else:
                    # Handle legacy parallel array format if needed
                    llm_dict["models"] = []
            
            # Handle evaluator_models field (current format)
            if "evaluator_models" in llm_dict and isinstance(llm_dict["evaluator_models"], list):
                # Check if it's a list of dictionaries (object format)
                if llm_dict["evaluator_models"] and isinstance(llm_dict["evaluator_models"][0], dict):
                    llm_dict["evaluator_models"] = [ModelConfig(**m) for m in llm_dict["evaluator_models"]]
                else:
                    # Handle legacy format if needed
                    llm_dict["evaluator_models"] = []
            
            # Handle legacy evolution_models field (backward compatibility)
            if "evolution_models" in llm_dict:
                if isinstance(llm_dict["evolution_models"], list) and llm_dict["evolution_models"]:
                    if isinstance(llm_dict["evolution_models"][0], dict):
                        llm_dict["models"] = [ModelConfig(**m) for m in llm_dict["evolution_models"]]
                llm_dict.pop("evolution_models")
            
            config.llm = LLMConfig(**llm_dict)
        
        if "prompt" in config_dict:
            config.prompt = PromptConfig(**config_dict["prompt"])
        if "island" in config_dict:
            config.island = IslandConfig(**config_dict["island"])

        if "rag" in config_dict:
            config.rag = RAGConfig.from_dict(config_dict["rag"])
        
       
        if config.random_seed is not None:
            for model in config.llm.models + config.llm.evaluator_models:
                if model.random_seed is None:
                    model.random_seed = config.random_seed
        
        if "evaluator" in config_dict:
            config.evaluator = EvaluatorConfig(**config_dict["evaluator"])
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary"""
        return {
            # General settings
            "resume": self.resume,
            "max_iterations": self.max_iterations,
            "checkpoint_interval": self.checkpoint_interval,
            "log_level": self.log_level,
            "log_dir": self.log_dir,
            "random_seed": self.random_seed,
            "diff_based_evolution": self.diff_based_evolution,
            "allow_full_rewrites": self.allow_full_rewrites,
            "max_code_length": self.max_code_length,
            "init_program_path": self.init_program_path,
            "evalutor_file_path": self.evalutor_file_path,
            "artifacts_base_path": self.artifacts_base_path,
            "artifact_size_threshold": self.artifact_size_threshold,
            "cleanup_old_artifacts": self.cleanup_old_artifacts,
            "artifact_retention_days": self.artifact_retention_days,
            "enable_artifacts": self.enable_artifacts,
            "archive_size": self.archive_size,
            "checkpoint": self.checkpoint,
            "use_rag": self.use_rag,
            "rag_doc_path": self.rag_doc_path,
            "vector_save_dir": self.vector_save_dir,
            # Component configurations
            "llm": {
                "models": [
                    {
                        "name": model.name,
                        "api_key": model.api_key,
                        "base_url": model.base_url,
                        "temperature": model.temperature,
                        "top_p": model.top_p,
                        "max_tokens": model.max_tokens,
                        "timeout": model.timeout,
                        "weight": model.weight,
                        "retries": model.retries,
                        "retry_delay": model.retry_delay,
                        "random_seed": model.random_seed,
                        "use_web": model.use_web,
                        "tavily_api_key": model.tavily_api_key,
                    }
                    for model in self.llm.models
                ],
                "evaluator_models": [
                    {
                        "name": model.name,
                        "api_key": model.api_key,
                        "base_url": model.base_url,
                        "temperature": model.temperature,
                        "top_p": model.top_p,
                        "max_tokens": model.max_tokens,
                        "timeout": model.timeout,
                        "weight": model.weight,
                        "retries": model.retries,
                        "retry_delay": model.retry_delay,
                        "random_seed": model.random_seed,
                        "use_web": model.use_web,
                        "tavily_api_key": model.tavily_api_key,
                    }
                    for model in self.llm.evaluator_models
                ],
            },
            "prompt": {
                "template_dir": self.prompt.template_dir,
                "system_message": self.prompt.system_message,
                "evaluator_system_message": self.prompt.evaluator_system_message,
                "num_top_programs": self.prompt.num_top_programs,
                "num_diverse_programs": self.prompt.num_diverse_programs,
                "use_template_stochasticity": self.prompt.use_template_stochasticity,
                "template_variations": self.prompt.template_variations,
                "language": self.prompt.language,
                # Note: meta-prompting features not implemented
                # "use_meta_prompting": self.prompt.use_meta_prompting,
                # "meta_prompt_weight": self.prompt.meta_prompt_weight,
            },
            "island": {
                "population_size": self.island.population_size,
                "num_islands": self.island.num_islands,
                "elite_selection_ratio": self.island.elite_selection_ratio,
                "exploration_ratio": self.island.exploration_ratio,
                "exploitation_ratio": self.island.exploitation_ratio,
                "feature_dimensions": self.island.feature_dimensions,
                "feature_bins": self.island.feature_bins,
                "migration_interval": self.island.migration_interval,
                "migration_rate": self.island.migration_rate,
                "random_seed": self.island.random_seed,
                "evolution_direction": self.island.evolution_direction,
                "time_of_meeting": self.island.time_of_meeting,
                "diversity_metric":self.island.diversity_metric,
            },
            "evaluator": {
                "timeout": self.evaluator.timeout,
                "max_retries": self.evaluator.max_retries,
                # Note: resource limits not implemented
                # "memory_limit_mb": self.evaluator_config.memory_limit_mb,
                # "cpu_limit": self.evaluator_config.cpu_limit,
                "cascade_evaluation": self.evaluator.cascade_evaluation,
                "cascade_thresholds": self.evaluator.cascade_thresholds,
                "parallel_evaluations": self.evaluator.parallel_evaluations,
                # Note: distributed evaluation not implemented
                # "distributed": self.evaluator_config.distributed,
                "use_llm_feedback": self.evaluator.use_llm_feedback,
                "llm_feedback_weight": self.evaluator.llm_feedback_weight,
            },
         
            "graph":{
                "parallel_generate": self.graph.parallel_generate,
                "checkpoint_path": self.graph.checkpoint_path,
            },
            "controller":{
                "resume": self.controller.resume,
            },
            "rag": self.rag.to_dict()
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def copy(self)->"Config":
        return Config(**self.to_dict())

def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or use defaults"""
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
        print(f"loaded config from {config_path},success")
    else:
        config = Config()
        print(f"loaded config from {config_path},failed")

    # Make the system message available to the individual models, in case it is not provided from the prompt sampler
    config.llm.update_model_params({"system_message": config.prompt.system_message})

    return config



if __name__ == "__main__":
    config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/test_config.yaml")
    print(config)