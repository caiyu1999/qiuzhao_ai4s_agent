'''
这个文件实现一个llm_config类 与llm相关的配置
'''
from typing import Optional,List,Dict,Any
from dataclasses import dataclass,field


@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    name: str = "gpt-4o-mini"
    model_provider: str = "openai"
    base_url: str = "https://api.chatanywhere.tech/v1"
    system_message: str = "You are a helpful assistant."
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5
    api_key: str = "sk-p99d7DFP8ICuPQj4mnKIURGGInimP8EpeSxDpBnkB2BUTVRf"
    random_seed: Optional[int] = None
    weight: float = 1.0 
    use_web:bool = False
    tavily_api_key:str = "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"
    language:str = "English"
    
    
    def to_dict(self)->Dict[str,Any]: 
        return {
            "name": self.name,
            "model_provider": self.model_provider,
            "base_url": self.base_url,
            "system_message": self.system_message,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "weight": self.weight,
            "use_web": self.use_web,
            "tavily_api_key": self.tavily_api_key,
            "language": self.language,
        }
        
        
    
    

@dataclass
class LLMConfig(ModelConfig):
    """Configuration for LLM models 可以包含任意个数的模型"""

    # API configuration
    api_base: List[str] = field(default_factory=lambda: ["https://api.openai.com/v1"])
    language: str = "English"

    # Generation parameters
    system_message: Optional[str] = field(default_factory=lambda: "system_message")
    temperature: List[float] = field(default_factory=lambda: [0.7])
    top_p: List[float] = field(default_factory=lambda: [0.95])
    max_tokens: List[int] = field(default_factory=lambda: [8192])

    # Request parameters
    timeout: List[int] = field(default_factory=lambda: [60])
    retries: List[int] = field(default_factory=lambda: [3])
    retry_delay: List[int] = field(default_factory=lambda: [5])

    # n-model configuration for evolution LLM ensemble
    models: List[ModelConfig] = field(default_factory=lambda: [ModelConfig()])
    # n-model configuration for evaluator LLM ensemble
    evaluator_models: List[ModelConfig] = field(default_factory=lambda: [ModelConfig()])

    def __post_init__(self):
        """Post-initialization to set up model configurations"""
        # If no evaluator models are defined, use the same models as for evolution
        # 如果没有定义evaluator_models，则使用models作为evaluator_models
        if not self.evaluator_models or len(self.evaluator_models) < 1:
            self.evaluator_models = self.models.copy()
        else:
            self.evaluator_models = [m for m in self.evaluator_models]
        
        if not self.models or len(self.models) < 1:
            self.models = [ModelConfig() for _ in range(len(self.evaluator_models))]
        else:
            self.models = [m for m in self.models]

        # Update models with shared configuration values
        shared_config = {
            "api_base": self.api_base[0] if self.api_base else "https://api.openai.com/v1",
            "api_key": self.api_key,
            "temperature": self.temperature[0] if self.temperature else 0.7,
            "top_p": self.top_p[0] if self.top_p else 0.95,
            "max_tokens": self.max_tokens[0] if self.max_tokens else 8192,
            "timeout": self.timeout[0] if self.timeout else 60,
            "retries": self.retries[0] if self.retries else 3,
            "retry_delay": self.retry_delay[0] if self.retry_delay else 5,
            "random_seed": self.random_seed,
            "language": self.language,
        }
        self.update_model_params(shared_config)

    def update_model_params(self, args: Dict[str, Any], overwrite: bool = False) -> None:
        """Update model parameters for all models"""
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)
                    
    @classmethod
    def from_dict(cls, llm_dict:Dict[str,Any]) -> "LLMConfig":
        """Create LLMConfig from a dictionary"""
        return cls(**llm_dict)
    
    
    
    def to_dict(self) -> Dict[str,Any]:
        """Convert LLMConfig to a dictionary"""
        pass 