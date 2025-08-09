from typing import Optional,Any,Dict,List
from dataclasses import dataclass , field, asdict 



@dataclass 
class EmbeddingsConfig:
    model:str = "text-embedding-3-small"
    api_key:str = ""
    base_url:str = ""
    
    chunk_size:int = 1000
    max_retries:int = 3
    timeout:float = 30.0
    skip_empty:bool = True
    tiktoken_enabled:bool = True
    allowed_special:List[str] = field(default_factory=lambda: ["<|endoftext|>", "<|startoftext|>"])
    disallowed_special:List[str] = field(default_factory=lambda: [])
    retry_min_seconds:int = 1
    retry_max_seconds:int = 60
    
    def to_dict(self)->Dict[str,Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, embeddings_dict:Dict[str,Any]) -> "EmbeddingsConfig":
        """Create EmbeddingsConfig from a dictionary"""
        return cls(**embeddings_dict)
    
    
    
@dataclass 
class LLMRagConfig:
    model:str = "gpt-4o-mini"
    api_key:str = ""
    base_url:str = ""
    temperature:float = 0.7
    top_p:float = 0.95
    max_tokens:int = 8192
    timeout:int = 60
    
    def to_dict(self)->Dict[str,Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, llm_rag_dict:Dict[str,Any]) -> "LLMRagConfig":
        """Create LLMRagConfig from a dictionary"""
        return cls(**llm_rag_dict)
    
    
@dataclass
class RAGConfig:
    """RAG配置类，包含嵌入模型和LLM配置"""
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    llm: LLMRagConfig = field(default_factory=LLMRagConfig)
    use_RAG: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "embeddings": self.embeddings.to_dict(),
            "llm": self.llm.to_dict(),
            "use_RAG": self.use_RAG
        }
    
    @classmethod
    def from_dict(cls, rag_dict: Dict[str, Any]) -> "RAGConfig":
        """从字典创建RAGConfig"""
        embeddings_config = EmbeddingsConfig.from_dict(rag_dict.get("embeddings", {}))
        llm_config = LLMRagConfig.from_dict(rag_dict.get("llm", {}))
        
        return cls(
            embeddings=embeddings_config,
            llm=llm_config,
            use_RAG=rag_dict.get("use_RAG", True)
        )
    
    
if __name__ == "__main__":
    from langchain.chat_models import init_chat_model
    test_llm_config = LLMRagConfig()
    test_llm = init_chat_model(
        **test_llm_config.to_dict()
    )
    
    print(test_llm.invoke("Hello, how are you?").content)
    
    