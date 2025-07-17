from langchain.chat_models import init_chat_model
from openevolve_graph.Config.config import Config
from openevolve_graph.Config.llm_config import ModelConfig
import random
import logging 

logger = logging.getLogger(__name__)


class LLMs:
    def __init__(self,config:Config):
        self.config = config
        self.model_config_list=self.config.llm.models
        self.models = [self.init_model(model_config) for model_config in self.model_config_list]
        self.weights = [model_config.weight for model_config in self.model_config_list]
        self.timeout = [model_config.timeout for model_config in self.model_config_list]
        self.retries = [model_config.retries for model_config in self.model_config_list]
        self.retry_delay = [model_config.retry_delay for model_config in self.model_config_list]
        self.use_web = [model_config.use_web for model_config in self.model_config_list]
        self.tavily_api_key = [model_config.tavily_api_key for model_config in self.model_config_list]
        self.language = [model_config.language for model_config in self.model_config_list]
        self.system_message = [model_config.system_message for model_config in self.model_config_list]
        
    def init_model(self,model_config:ModelConfig):
        model = init_chat_model(
            model=model_config.name,
            model_provider=model_config.model_provider,
            api_key=model_config.api_key,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            max_tokens=model_config.max_tokens,
            timeout=model_config.timeout,
            base_url=model_config.base_url,
            )
        return model
        
    def sample_model(self):
        logger.info(f"Sampling model with weights: {self.weights}")
        return random.choices(self.models,weights=self.weights,k=1)[0]
    def invoke(self,prompt:str)->str:
        model = self.sample_model()
        return model.invoke(prompt).content
        
        
        
class LLMs_evalutor():
    def __init__(self,config:Config):
        self.config = config
        self.model_config_list=self.config.llm.evaluator_models
        self.models = [self.init_model(model_config) for model_config in self.model_config_list]
        self.weights = [model_config.weight for model_config in self.model_config_list]
        self.timeout = [model_config.timeout for model_config in self.model_config_list]
        self.retries = [model_config.retries for model_config in self.model_config_list]
        self.retry_delay = [model_config.retry_delay for model_config in self.model_config_list]
        self.use_web = [model_config.use_web for model_config in self.model_config_list]
        self.tavily_api_key = [model_config.tavily_api_key for model_config in self.model_config_list]
        self.language = [model_config.language for model_config in self.model_config_list]
        self.system_message = [model_config.system_message for model_config in self.model_config_list]
    def init_model(self,model_config:ModelConfig):
        model = init_chat_model(
            model=model_config.name,
            model_provider=model_config.model_provider,
            api_key=model_config.api_key,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            max_tokens=model_config.max_tokens,
            timeout=model_config.timeout,
            base_url=model_config.base_url,
            )
        return model
    
    def sample_model(self):
        return random.choices(self.models,weights=self.weights,k=1)[0]
    def invoke(self,prompt:str)->str:
        model = self.sample_model()
        return model.invoke(prompt).content
        
        
        
        
        

if __name__ == "__main__":
    config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
    llms = LLMs(config)
    print(llms.invoke("Hello, how are you?"))

    
        
        
        
        
        
        
        