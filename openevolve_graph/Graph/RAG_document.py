import asyncio 
from typing import Optional,Any, Dict
import os
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,CSVLoader
from dataclasses import dataclass ,field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4
from langchain_community.vectorstores import FAISS
from openevolve_graph.Config.rag_config import RAGConfig
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain.chains.summarize import load_summarize_chain

import logging 
logger = logging.getLogger(__name__)




# 文档类 
@dataclass 
class document:
    '''
    文档类 
    为每个文档创建独立的向量存储
    '''
    file_path: str = ""  # 文档路径
    file_type: str = "pdf"  # 文档类型
    vector_store_path: str = ""  # 向量存储路径
    pages: list[Any] = field(default_factory=list)  # 文档分片
    len_pages: int = 0  # 分片数量
    # vector_store: Optional[FAISS] = None  # 向量存储
    RAG_config: RAGConfig = field(default_factory=RAGConfig)
    id:str = str(uuid4())# 生成一个自己的独立ID 
    vector_save_dir:str = ""
    summary: str = ""  # 文档摘要
    summary_generated: bool = False  # 是否已生成摘要
    chunk_size:int = 500
    chunk_overlap:int = 100
    
    # 用于存储额外的可变参数
    _extra_params: Dict[str, Any] = field(default_factory=dict, init=False)
    def __post_init__(self):
        """同步初始化，内部调用异步逻辑"""
        self.embeddings = OpenAIEmbeddings(**self.RAG_config.embeddings.to_dict())
        self.llm = init_chat_model(**self.RAG_config.llm.to_dict())
        # self.file_type = self.file_path.split(".")[-1]
        self.vector_store_path = f"{self.vector_save_dir}/{self.id}"
        
        # print(self.file_type,"file_type")
    async def initialize(self):
            # print(self.file_type,"file_type")
            """异步初始化文档"""
            self.pages = await self._load_documents()
            self.len_pages = len(self.pages)
            self.create_vector_store()
            await self.generate_summary()
    async def _load_documents(self):
        """异步加载文档的辅助方法"""
        if self.file_type == "pdf":
            return await self.load_pdf(self.chunk_size, self.chunk_overlap)
        elif self.file_type == "docx":
            return await self.load_docx(self.chunk_size, self.chunk_overlap)
        elif self.file_type == "csv":
            return await self.load_csv(self.chunk_size, self.chunk_overlap)
        else:
            raise ValueError(f"不支持的文件类型: {self.file_type}")
    
    @classmethod
    def create_with_extra_params(cls, file_path: str, **kwargs) -> "document":
        """使用可变参数创建文档实例"""
        # 提取dataclass字段
        dataclass_fields = {
            'file_path': file_path,
            'file_type': kwargs.get('file_type', '.pdf'),
            'vector_store_path': kwargs.get('vector_store_path', ''),
            'embeddings_config': kwargs.get('embeddings_config', RAGConfig())
        }
        
        # 创建实例
        instance = cls(**dataclass_fields)
        
        # 存储额外的参数
        extra_params = {k: v for k, v in kwargs.items() 
                       if k not in dataclass_fields}
        instance._extra_params = extra_params
        
        return instance
    
    def get_extra_param(self, key: str, default: Any = None) -> Any:
        """获取额外参数"""
        return self._extra_params.get(key, default)
    
    def set_extra_param(self, key: str, value: Any) -> None:
        """设置额外参数"""
        self._extra_params[key] = value
    
    def create_vector_store(self):
        vector_store = None
        """为文档创建向量存储"""
        if os.path.exists(self.vector_store_path):
            # 如果已存在，直接加载
            vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True  # 允许加载本地文件
            )
        else:
            # 创建新的向量存储
            vector_store = FAISS.from_documents(
                self.pages, 
                self.embeddings
            )
            # 保存到本地
            os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
            vector_store.save_local(self.vector_store_path)
    
    def split_pages(self, pages , chunk_size:int = 500, chunk_overlap:int = 100):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(pages)
        uuids = [str(uuid4()) for _ in range(len(chunks))] # 为每一个分片生成一个自己的uuid 
        for chunk, uuid in zip(chunks, uuids):
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = {}
            chunk.metadata["uuid"] = uuid
            chunk.metadata["source_file"] = self.file_path
        return chunks 
    
    async def load_pdf(self,chunk_size:int = 500,chunk_overlap:int = 100)->list[Any]:
        pages = []
        loader = PyPDFLoader(self.file_path)
        async for page in loader.alazy_load():
            pages.append(page)
        return self.split_pages(pages,chunk_size,chunk_overlap)
    
    async def load_docx(self,chunk_size:int = 500,chunk_overlap:int = 100)->list[Any]:
        pages = []
        loader = Docx2txtLoader(self.file_path)
        async for page in loader.alazy_load():
            pages.append(page)
        return self.split_pages(pages,chunk_size,chunk_overlap)
    
    async def load_csv(self,chunk_size:int = 500,chunk_overlap:int = 100)->list[Any]:
        pages = []
        loader = CSVLoader(self.file_path)
        async for page in loader.alazy_load():
            pages.append(page)
        return self.split_pages(pages,chunk_size,chunk_overlap)


    async def generate_summary(self) -> str:
        if not self.pages:
            return "空文档"

        # 动态选择链类型
        total_length = sum(len(d.page_content) for d in self.pages)
        chain_type = "map_reduce" if total_length > 4000 else "stuff"

        if chain_type == "map_reduce":
            # Map-Reduce 专用提示模板
            map_prompt = PromptTemplate.from_template("""
                Please extract 1-2 key points from the following text segment (describe in short sentences):
                {text}
                Key points:
            """)
            combine_prompt = PromptTemplate.from_template("""
                Based on the following key points, generate a final summary. Make sure to include each key point, be concise and clear, do not exceed 500 words, and do not include content from other documents:
                {text}
                Summary:
            """)
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,      # Map prompt for chunk processing
                combine_prompt=combine_prompt,  # Combine prompt
                verbose=False
            )
        else:
            # Stuff chain prompt template
            prompt = PromptTemplate.from_template("""
                Please generate a concise summary (no more than 300 words) for the following document:
                {text}
                Summary:
            """)
            chain = load_summarize_chain(
                self.llm,
                chain_type="stuff",
                prompt=prompt,
                verbose=True
            )

        self.summary = await chain.arun(self.pages)
        return self.summary
    def _extract_key_info(self) -> str:
            """提取关键信息生成摘要"""
            if not self.pages:
                return "空文档"
            
            # 提取文件名
            file_name = os.path.basename(self.file_path)
            
            # 提取文档类型和页数信息
            doc_info = []
            doc_info.append(f"文件名: {file_name}")
            doc_info.append(f"类型: {self.file_type}")
            doc_info.append(f"分片数: {self.len_pages}")
            
            # 提取前几个分片的关键词
            if self.pages:
                first_chunk = self.pages[0].page_content if hasattr(self.pages[0], 'page_content') else str(self.pages[0])
                # 提取前100个字符作为预览
                preview = first_chunk[:100].replace('\n', ' ').strip()
                if preview:
                    doc_info.append(f"内容预览: {preview}")
            
            return " | ".join(doc_info)
    
    
    def get_document_info(self) -> Dict[str, Any]:
        """获取文档信息"""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "len_pages": self.len_pages,
            "summary": self.summary,
            "vector_store_path": self.vector_store_path,
        }
        
        
    async def search(self, question: str, k: int = 5):
        '''
        针对文档内容搜索一个问题
        返回相关的文档片段
        '''
        # 首先获取存储的向量 
        vector_store = FAISS.load_local(
            self.vector_store_path, 
            self.embeddings,
            allow_dangerous_deserialization=True  # 允许加载本地文件
        )
        
        # similarity_search返回的是Document对象列表
        similarity_results = vector_store.similarity_search(question, k=k)
        
        # 提取文本内容和元数据
        search_results = []
        for i, doc in enumerate(similarity_results):
            result = {
                "rank": i + 1,
                "content": doc.page_content,  # 文档内容（文本）
                "metadata": doc.metadata,     # 元数据（字典）
                "source_file": doc.metadata.get("source_file", "unknown"),
                "uuid": doc.metadata.get("uuid", "unknown"),
                "question": question
            }
            search_results.append(result)
        
        return search_results
    
    def search_simple(self, question: str, k: int = 5) -> str:
        '''
        简化版搜索，直接返回文本内容
        '''
        vector_store = FAISS.load_local(
            self.vector_store_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        similarity_results = vector_store.similarity_search(question, k=k)
        
        # 只返回文本内容
        texts = [doc.page_content for doc in similarity_results]
        return "\n\n---\n\n".join(texts)
        
        
        

if __name__ == "__main__":
    config_path ="/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/test_config.yaml"
    from openevolve_graph.Config.config import Config 
    config = Config.from_yaml(config_path) 
    file_path = "/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/documents/过度参数化神经网络 Hessian 矩阵的实证分析.pdf"
    doc = document(
        file_path=file_path,
        RAG_config=config.rag,
        vector_save_dir="/Users/caiyu/Desktop/langchain/openevolve_graph/circle_packing/documents/vector_store"
    )
    

    