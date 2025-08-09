# 这里的RAG作为一个节点 这个节点在初始化后进行 在
# 首先 检查文件夹中的文档类型(目前只支持docx pdf csv格式)
import asyncio 
from typing import Optional,Any, Dict,List
import os

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
from openevolve_graph.Config.config import Config
import time 
import logging 
logger = logging.getLogger(__name__) 

class RAG_output(BaseModel):
   documents_ids:list[str] = Field(description="The ids of the documents that need to be searched",default_factory=list)
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
        5. rag存储库中存储了哪些信息
        这个prompt要求llm生成它认为现在需要进行rag来获得的一些信息
    3.llm会返回它认为的需要检索的信息
    4.检索这些信息 并返回相关信息
    5.让llm总结相关信息 并返回总结 
    6.在下一次演化中 这个总结将会被添加到提示词中来增强代码的生成

    '''
    def __init__(self,
                 config:Config):
        self.RAG_config = config.rag
        self.embeddings_config = self.RAG_config.embeddings
        
        self.llm_config = self.RAG_config.llm
        self.llm = init_chat_model(**self.llm_config.to_dict())
        self.embeddings = OpenAIEmbeddings(**self.embeddings_config.to_dict())
        self.llm = self.llm.with_structured_output(RAG_output)
        self.retry_times = 3 
        self.retry_delay = 1 
        
    async def execute(self, GraphState: GraphState):
        '''
        根据需求去文档中搜索相关信息
        在整理格式后返回相关信息 如果出错或者无返回 则不改变任何信息
        '''
        
        status , GraphState =await self.check_documents(GraphState) #用于更新文档
        if status == RAG_dir_status.DIR_NOT_EXIST:
            logger.warning("Document directory does not exist")
            return GraphState
        elif status == RAG_dir_status.EMPTY:
            logger.warning("Document directory is empty")
            return GraphState
        elif status == RAG_dir_status.NO_CHANGE:
            logger.warning("Document directory is not changed")
            return GraphState
        
        
        # 生成提示词 并更新raginfo
        prompt = self.generate_prompt(GraphState)
        print("prompt done",prompt)
        
        
        
        try:
            response = self.llm.invoke(prompt)
            
        except Exception as e:
            for i in range(self.retry_times):
                time.sleep(self.retry_delay)
                try:
                    response = self.llm.invoke(prompt)
                    break
                except Exception as e:
                    logger.warning(f"Failed to generate prompt {e}")
                    continue
            if response is None or response =={}:
                return 
        
        
        print(response)
        if response.documents_ids ==[] or response.questions ==[]:
            return GraphState
        
        questions = response.questions
        documents_ids = response.documents_ids
        if len(documents_ids) != len(questions):
            logger.warning("The length of documents_ids and questions is not the same")
            return GraphState
        
        response ={documents_ids[i]:[] for i in range(len(questions))}
        for i in range(len(questions)):
            response[documents_ids[i]].append(questions[i])
        
        
        
        if prompt =={} or prompt ==None:
            logger.warning("No questions to ask") 
            return GraphState
        
        else:
            rag_help_info = {}
            for document_id,questions in response.items():
                try:
                    doc = GraphState.Documents[document_id]
                    results:list[Any] = await asyncio.gather(*[doc.search(question) for question in questions])# 搜索得到的信息 results中的每一个元素都是list[dict] 
                except Exception as e:
                    logger.warning(f"Failed to search document {document_id}")
                    continue
                # 将结果进行整理 
                for result in results:
                    #result是一个搜索得到的相关信息 需要将它进行整理             
                    # result = {
                    #     "rank": i + 1,
                    #     "content": doc.page_content,  # 文档内容（文本）
                    #     "metadata": doc.metadata,     # 元数据（字典）
                    #     "source_file": doc.metadata.get("source_file", "unknown"),
                    #     "uuid": doc.metadata.get("uuid", "unknown")
                    # }
                    rag_help_info[result[0]['question']] = []
                    for answer in result:
                        #首先对answer的格式进行调整
                        rag_help_info[result[0]['question']].append(answer['content'])#将有关信息都添加到rag_help_info中
                        
        # 更新GraphState中的RAG_help_info
        # 首先整理为str格式
        rag_help_info_str = ""
        for question,answers in rag_help_info.items():
            rag_help_info_str += f"问题: {question}\n"
            for answer in answers:
                rag_help_info_str += f"答案: {answer}\n"
            rag_help_info_str += "\n"
        
        
        
        
        
        GraphState.RAG_help_info = rag_help_info_str 
        print(rag_help_info)
        # 更新每一个岛屿中的RAG_help_info 
        for island_id,island_state in GraphState.islands.items():
            island_state.RAG_help_info = rag_help_info_str
        return GraphState           
    async def check_documents(self,GraphState:GraphState):
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
                doc_type = file.split(".")[-1]
                if doc_type not in ["pdf","docx","csv"]:
                    continue
                if file not in previous_rag_list: #此时需要添加 
                   
                   
                   vector_save_dir = GraphState.vector_save_dir
                #    print(doc_type,"doc_type")
                   doc = document(file_path=os.path.join(doc_dir_path, file),
                                  file_type=doc_type,
                                  vector_save_dir=vector_save_dir,
                                  RAG_config=self.RAG_config)
                   
                   await doc.initialize() 
                   doc_id = doc.id 
                   # 添加新的key和value
                   GraphState.Documents[doc_id] = doc 
                   # 更新新的文件的地址
                   GraphState.rag_doc_list.append(file)
                   GraphState.Documents_abstract[doc_id] = doc.summary
                #    print(GraphState.Documents_abstract)
                   GraphState.vector_save_dir = doc.vector_save_dir
                   logger.info(f"New document added: {doc_id}")
                   
                   
            for file in previous_rag_list:
                if file not in file_list:# 此时需要删除
                    doc_id = GraphState.Documents[file].id
                    del GraphState.Documents[doc_id]
                    GraphState.rag_doc_list.remove(file)
                    GraphState.Documents_abstract.pop(doc_id)
                    logger.info(f"Document removed: {doc_id}")
        
        return status,GraphState 
    
    def generate_prompt(self, GraphState:GraphState):
        '''
        Generate prompt
        '''
        message = ChatPromptTemplate.from_template('''
        You are an experienced code evolution expert. Based on the following requirements, and by leveraging the content of the professional knowledge base, retrieve professional knowledge that can improve the performance or accuracy of the current code.
        The current best program code is as follows:
        {best_program_code}

        Key information summaries of each document in the professional knowledge base are as follows:
        {Documents_abstract}
        
        Please return two lists:
        List 1: The IDs of the documents to be retrieved (one ID corresponds to one question; if you have multiple questions for the same document, add the same document ID multiple times in List 1)
        List 2: The questions to be retrieved (one ID corresponds to one question; if you have multiple questions for the same document, add the same question multiple times in List 2)
        
        Note: Each document ID corresponds to a document question. If you have multiple questions for the same document, add the same document ID multiple times in List 1, and add the same question multiple times in List 2.
    
        Example return format (for a certain question). Note that the lengths of the two lists must be the same:
        List 1: ["Document1", "Document2", "Document2", "Document1"]
        List 2: ["What is the core innovation of this algorithm?", "What is the dimension of the data?", "What is the detailed process of the algorithm?", "Are there any precautions?"]
        
        If you do not know what to retrieve, please return two empty lists.
        ''')

        abstract = {key: value.summary for key, value in GraphState.Documents.items()}

        return message.invoke({
            "best_program_code": GraphState.best_program.code,
            "Documents_abstract": abstract
        })



if __name__ == "__main__":
    # 测试代码
    path = "/Users/caiyu/Desktop/langchain/openevolve_graph/过度参数化神经网络 Hessian 矩阵的实证分析.pdf"
    doc = document(file_path=path)
    print(doc.pages[0].metadata)
    print(len(doc.pages))

    
    
    
    
    
    
    
    
    