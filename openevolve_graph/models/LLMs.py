from langchain.chat_models import init_chat_model
from openevolve_graph.Config.config import Config
from openevolve_graph.Config.llm_config import ModelConfig
import random
from typing import Optional, Union, TypeVar, Type, List, Dict, Any
import logging 
from pydantic import BaseModel, Field
logger = logging.getLogger(__name__)
import asyncio


class LLMs:
    def __init__(self, config: Config):
        self.config = config
        self.model_config_list = self.config.llm.models
        self.models = [self.init_model(model_config) for model_config in self.model_config_list]
        self.weights = [model_config.weight for model_config in self.model_config_list]
        self.timeout = [model_config.timeout for model_config in self.model_config_list][0]  # 这里后面要修改
        self.retries = [model_config.retries for model_config in self.model_config_list][0]
        self.retry_delay = [model_config.retry_delay for model_config in self.model_config_list][0]
        self.use_web = [model_config.use_web for model_config in self.model_config_list][0]
        self.tavily_api_key = [model_config.tavily_api_key for model_config in self.model_config_list][0]
        self.language = [model_config.language for model_config in self.model_config_list][0]
        self.system_message = [model_config.system_message for model_config in self.model_config_list][0]
        
    def init_model(self, model_config: ModelConfig):
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
        return random.choices(self.models, weights=self.weights, k=1)[0]

    async def call_api(self, model, prompt):
        """带超时和重试机制的API调用"""
        for i in range(self.retries):
            try:
                # 添加超时控制
                result = await asyncio.wait_for(
                    model.ainvoke(prompt), 
                    timeout=self.timeout
                )
                # 若成功生成 则返回结果
                return result
            
            except asyncio.TimeoutError:
                # 若超时 则重试
                logger.error(f"Timeout after {self.timeout} seconds on attempt {i+1}")
                if i < self.retries - 1:  # 不是最后一次尝试
                    await asyncio.sleep(self.retry_delay)
                    continue 
                else:
                    raise TimeoutError(f"Model invocation timed out after {self.retries} attempts")
            except Exception as e:
                # 若失败 则重试
                logger.error(f"Error invoking model on attempt {i+1}: {e}")
                if i < self.retries - 1:  # 不是最后一次尝试
                    await asyncio.sleep(self.retry_delay)
                    continue 
                else:
                    raise
        
    async def invoke(self, prompt: str, 
                     structure: Optional[Type[BaseModel] | None], 
                     key: Optional[Union[str, List[str]] | None]
                     ) -> Union[str, BaseModel, None, Dict[str, Any]]:
        model = self.sample_model()
        if structure is not None:
            model = model.with_structured_output(structure)
            if key is not None:
                try:
                    result = await self.call_api(model, prompt)
                    if result is not None and isinstance(key, list):
                        return {k: getattr(result, k) for k in key}
                    elif result is not None and isinstance(key, str):
                        return getattr(result, key)
                    return None
                except Exception as e:
                    logger.error(f"Error invoking model: {e}")
                    return None
            else:
                try:
                    result = await self.call_api(model, prompt)
                    return result
                except Exception as e:
                    logger.error(f"Error invoking model: {e}")
                    return None
        else:
            try:
                result = await self.call_api(model, prompt)
                if result is not None and hasattr(result, 'content'):
                    return result.content
                return str(result) if result is not None else None
            except Exception as e:
                logger.error(f"Error invoking model: {e}")
                return None


class LLMs_evalutor():
    def __init__(self, config: Config):
        self.config = config
        self.model_config_list = self.config.llm.evaluator_models
        self.models = [self.init_model(model_config) for model_config in self.model_config_list]
        self.weights = [model_config.weight for model_config in self.model_config_list]
        self.timeout = [model_config.timeout for model_config in self.model_config_list]
        self.retries = [model_config.retries for model_config in self.model_config_list]
        self.retry_delay = [model_config.retry_delay for model_config in self.model_config_list]
        self.use_web = [model_config.use_web for model_config in self.model_config_list]
        self.tavily_api_key = [model_config.tavily_api_key for model_config in self.model_config_list]
        self.language = [model_config.language for model_config in self.model_config_list]
        self.system_message = [model_config.system_message for model_config in self.model_config_list]
    
    def init_model(self, model_config: ModelConfig):
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
        return random.choices(self.models, weights=self.weights, k=1)[0]
    def init_all_models(self):
        return [self.init_model(model_config) for model_config in self.model_config_list]
    
    async def call_api(self, model, prompt):
        """带超时和重试机制的API调用 因为并行评估 重试后如果结果中个数>=1即可"""
        for i in range(self.retries[0]):  # 使用第一个模型的配置
            try:
                # 添加超时控制
                result = await asyncio.wait_for(
                    model.ainvoke(prompt), 
                    timeout=self.timeout[0]  # 使用第一个模型的超时配置
                )
                return result
            except asyncio.TimeoutError:
                logger.error(f"Timeout after {self.timeout[0]} seconds on attempt {i+1}")
                if i < self.retries[0] - 1:  # 不是最后一次尝试
                    await asyncio.sleep(self.retry_delay[0])
                else:
                    return None
            except Exception as e:
                logger.error(f"Error invoking model on attempt {i+1}: {e}")
                if i < self.retries[0] - 1:  # 不是最后一次尝试
                    await asyncio.sleep(self.retry_delay[0])
                else:
                    return None
                    
    async def _invoke_single(self,model,prompt):
        return await self.call_api(model,prompt)
    
    async def invoke_parallel(self,
                              prompt:str,
                              structure:Optional[Type[BaseModel] | None],
                              key:Optional[Union[str, List[str]] | None])->Optional[str|BaseModel|None|Dict[str, Any]|List[BaseModel|str]]:
        '''
        并行评估 传入一个prompt 返回每一个模型的结果 
        ''' 
        models = self.init_all_models() 
        if structure is not None:
            models = [model.with_structured_output(structure) for model in models]
            if key is not None:
                try: #并行invoke
                    results = await asyncio.gather(*[(self._invoke_single(model,prompt)) for model in models])
                    # 检查results中非None的 个数 如果>=1 则返回结果即可 否则报错
                    results = [result for result in results if result is not None]
                    if len(results) == 0:
                        raise ValueError("No results returned from models")
                    
                    if key is not None:
                        if isinstance(key, list):
                            #若key为list 则返回一个dict  每一个key是要求的key value是这个key的list
                            return {k: [getattr(result, k) for result in results] for k in key}
                        elif isinstance(key, str):
                            #若key为str 同理
                            return {k:[getattr(result, k) for result in results] for k in key}

                except Exception as e:
                    logger.error(f"Error invoking model: {e}")
                    return None
            else:#若没有key 则返回每一个模型的输出即可 这里的每一个输出都是一个BaseModel
                try:
                    results = await asyncio.gather(*[(self._invoke_single(model,prompt)) for model in models])
                    results = [result for result in results if result is not None]
                    if len(results) == 0:
                        raise ValueError("No results returned from models")
                    return results
                except Exception as e:
                    logger.error(f"Error invoking model: {e}")
                    return None
        else:
            try:#如果structure为None 则返回每一个模型的输出即可 这里的每一个输出都是一个str
                results = await asyncio.gather(*[(self._invoke_single(model,prompt)) for model in models])
                results = [result for result in results if result is not None]
                if len(results) == 0:
                    raise ValueError("No results returned from models")
                return results
            except Exception as e:
                logger.error(f"Error invoking model: {e}")
                return None
        
    async def invoke(self, prompt: str, structure: Optional[Type[BaseModel] | None], key: Optional[Union[str, List[str]] | None]) -> Union[str, BaseModel, None, Dict[str, Any]]:
        model = self.sample_model()
        if structure is not None:
            model = model.with_structured_output(structure)
            if key is not None:
                try:
                    result = await self.call_api(model, prompt)
                    if result is not None and isinstance(key, list):
                        return {k: getattr(result, k) for k in key}
                    elif result is not None:
                        return getattr(result, key)
                    return None
                except Exception as e:
                    logger.error(f"Error invoking model: {e}")
                    return None
            else:
                try:
                    result = await self.call_api(model, prompt)
                    return result
                except Exception as e:
                    logger.error(f"Error invoking model: {e}")
                    return None
        else:
            try:
                result = await self.call_api(model, prompt)
                if result is not None and hasattr(result, 'content'):
                    return result.content
                return str(result) if result is not None else None
            except Exception as e:
                logger.error(f"Error invoking model: {e}")
                return None


# if __name__ == "__main__":
    # config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
    # llms = LLMs(config)
    # class Test(BaseModel):  
    #     answer: str = Field(default="", description="The answer to the question")
    #     age: int = Field(default=0, description="The age of the person")
        
    # print(asyncio.run(llms.invoke("Hello, how are you?", Test, ["answer", "age"])))
    # list_a = []
    # list_a.append(None)
    # print(list_a)        
    # test_dict = {"a":10}
    # print(test_dict.a)
        
        
        
        
        
        