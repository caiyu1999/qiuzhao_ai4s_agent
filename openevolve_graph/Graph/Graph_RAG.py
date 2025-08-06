# 这里的RAG作为一个节点 这个节点在初始化后进行 在
# 首先 检查文件夹中的文档类型(目前只支持docx pdf csv格式)
import asyncio 
from typing import Optional,Any, Dict
import os
import faiss 
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,CSVLoader
from dataclasses import dataclass ,field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4
from langchain_community.vectorstores import FAISS
from openevolve_graph.Config.rag_config import EmbeddingsConfig, LLMRagConfig
from openevolve_graph.Graph.Graph_Node_ABC import SyncNode
from openevolve_graph.Graph.Graph_state import GraphState
from langchain.chat_models import init_chat_model
from enum import Enum
from openevolve_graph.Graph.RAG_document import document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field 




class RAG_output(BaseModel):
    '''
    第一次传给llm所返回的需要搜索的信息列表
    '''
    questions:list[str] = Field(description="The questions that need to be asked to the documents",default_factory=list)
    



class RAG_dir_status(Enum):
    EMPTY = "empty" # 空文件夹
    UPDATED = "updated" # 更新了
    NO_CHANGE = "no_change" # 没有变化
    DIR_NOT_EXIST = "dir_not_exist" # 文件夹不存在


# RAG节点
class node_rag(SyncNode):
    '''
    这个节点在每一次meeting后进行 
    它会根据需求去文档中搜索相关信息
    在整理格式后返回相关信息
    注意 查询的时候是基于GraphState中的docmunet 而不是文件夹 
    
    
    1.首先它会先检查是否更新了矢量存储 如果有新的文件在文件夹中(或者旧的文件被删除) 则会更新矢量存储库
    2.生成提示词 包含:
        1. 曾经的代码演化历程 (简要)
        2. 曾经的代码演化出现的错误
        3. 曾经的代码演化的进步得益于哪些提升
        4. 当前的最佳代码
        5. rag存储库中存储了哪些信息
        这个prompt要求llm生成它认为现在需要进行rag来获得的一些信息
    3.llm会返回它认为的需要检索的信息
    4.检索这些信息 并返回相关信息
    5.让llm总结相关信息 并返回总结 
    6.在下一次演化中 这个总结将会被添加到提示词中来增强代码的生成

    '''
    def __init__(self,
                 embeddings_config: EmbeddingsConfig,
                 llm_config: LLMRagConfig):
        self.embeddings_config = embeddings_config
        self.llm_config = llm_config
        self.llm = init_chat_model(**self.llm_config.to_dict())
        self.embeddings = OpenAIEmbeddings(**self.embeddings_config.to_dict())
        self.documents = {}
        
    def execute(self, GraphState: GraphState):
        '''
        根据需求去文档中搜索相关信息
        在整理格式后返回相关信息
        '''
        pass
        
    def check_documents(self,GraphState:GraphState):
        '''
        检查文档是否更新
        如果更新了 则更新文档
        如果删除了 则删除文档
        '''
        
        # 首先检查文件夹中的档案是否更新
        
        status = None 
        doc_dir_path = GraphState.rag_doc_path 
        previous_rag_list = GraphState.rag_doc_list
        
        if not os.path.exists(doc_dir_path):
            status = RAG_dir_status.DIR_NOT_EXIST
        
        file_list = os.listdir(doc_dir_path) 
        if file_list == GraphState.rag_doc_list:
            status = RAG_dir_status.NO_CHANGE
        elif file_list == []:
            status = RAG_dir_status.EMPTY
        else:
            status = RAG_dir_status.UPDATED
            
            
            
        # 如果更新了文档 则要重新建立矢量存储
        if status == RAG_dir_status.UPDATED:
           # 首先检查是否删除了某些文件 如果删除了 则要删除对应的矢量存储 
            for file in file_list:
               if file not in previous_rag_list: #此时需要添加 
                   doc_type = file.split(".")[-1]
                   vector_save_dir = GraphState.vector_save_dir
                   
                   doc = document(file_path=os.path.join(doc_dir_path, file),
                                  file_type=doc_type,
                                  vector_save_dir=vector_save_dir,
                                  embeddings_config=self.embeddings_config)
                   
                   doc_id = doc.id 
                   # 添加新的key和value
                   GraphState.Documents[doc_id] = doc 
                   # 更新新的文件的地址
                   GraphState.rag_doc_list.append(file)
            for file in previous_rag_list:
                if file not in file_list:# 此时需要删除
                    doc_id = GraphState.Documents[file].id
                    del GraphState.Documents[doc_id]
                    GraphState.rag_doc_list.remove(file)
        
        return status , GraphState 
    
    def generate_prompt(self,GraphState:GraphState):
        '''
        生成提示词
        '''
        message = ChatPromptTemplate.from_template('''
        You are an experienced code evolution expert. According to the requirements, you need to search the documents for professional knowledge that can improve the code's performance or accuracy.
        The evolution history of this code is as follows: {evolution_history},
        The problems encountered during its evolution are as follows: {evolution_problem},
        The progress made during its evolution is as follows: {evolution_progress},
        The best code achieved during its evolution is as follows: {evolution_best_code},
        The current information in the professional knowledge base is as follows: {Documents_abstract},
        Please tell me what questions you think should be asked to improve the accuracy and performance of the code.
        Please return a list, where each item is a string.
        Example (for a certain problem):
        ["What is the data dimension?", "What is the output value in 2020?"]
        ''')
        
        
        
        
        
        
    


    


if __name__ == "__main__":
    # 测试代码
    path = "/Users/caiyu/Desktop/langchain/openevolve_graph/过度参数化神经网络 Hessian 矩阵的实证分析.pdf"
    doc = document(file_path=path)
    print(doc.pages[0].metadata)
    print(len(doc.pages))

    
    
    
    
    
    
    
    
    