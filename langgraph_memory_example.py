#!/usr/bin/env python3
"""
LangGraph本地内存保存示例
基于LangGraph官方文档的最佳实践
"""

import os
import sqlite3
import uuid
from typing import Dict, List, Any, Annotated
from typing_extensions import TypedDict
from datetime import datetime

# 1. 安装必要的依赖 (运行前请执行)
# pip install langgraph langgraph-checkpoint-sqlite

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

# 2. 定义图状态
class ConversationState(TypedDict):
    """对话状态定义"""
    messages: Annotated[List[Dict[str, Any]], add_messages]
    user_id: str
    conversation_context: Dict[str, Any]

# 3. 创建持久化组件
class LocalMemoryManager:
    """本地内存管理器"""
    
    def __init__(self, db_path: str = "langgraph_memory.db"):
        """
        初始化本地内存管理器
        
        Args:
            db_path: SQLite数据库路径
        """
        self.db_path = db_path
        
        # 创建SQLite检查点保存器 (线程级持久化)
        self.checkpointer = SqliteSaver.from_conn_string(db_path)
        
        # 创建内存存储器 (跨线程共享信息)
        self.store = InMemoryStore()
        
        print(f"✅ 本地内存管理器已初始化")
        print(f"📁 数据库路径: {db_path}")
        print(f"🔧 检查点保存器: {type(self.checkpointer).__name__}")
        print(f"📦 存储器: {type(self.store).__name__}")
    
    def save_user_memory(self, user_id: str, memory_type: str, memory_data: Dict[str, Any]):
        """
        保存用户记忆到跨线程存储
        
        Args:
            user_id: 用户ID
            memory_type: 记忆类型 (如 "preferences", "facts", "history")
            memory_data: 记忆数据
        """
        namespace = (user_id, memory_type)
        memory_id = str(uuid.uuid4())
        
        # 添加时间戳
        memory_with_timestamp = {
            **memory_data,
            "saved_at": datetime.now().isoformat(),
            "memory_id": memory_id
        }
        
        self.store.put(namespace, memory_id, memory_with_timestamp)
        print(f"💾 已保存用户记忆: {user_id} - {memory_type}")
        return memory_id
    
    def get_user_memories(self, user_id: str, memory_type: str) -> List[Dict[str, Any]]:
        """
        获取用户记忆
        
        Args:
            user_id: 用户ID
            memory_type: 记忆类型
        
        Returns:
            List[Dict]: 用户记忆列表
        """
        namespace = (user_id, memory_type)
        memories = self.store.search(namespace)
        return [memory.value for memory in memories]
    
    def get_conversation_state(self, thread_id: str) -> Dict[str, Any]:
        """
        获取对话状态
        
        Args:
            thread_id: 线程ID
        
        Returns:
            Dict: 对话状态
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            # 这里需要一个已编译的图才能获取状态
            # 在实际使用时，你需要传入已编译的图
            return {"状态": "需要在已编译的图上调用get_state()"}
        except Exception as e:
            print(f"⚠️ 获取对话状态失败: {e}")
            return {}
    
    def list_all_threads(self) -> List[str]:
        """
        列出所有线程ID
        
        Returns:
            List[str]: 线程ID列表
        """
        try:
            # 查询数据库中的所有线程
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            threads = [row[0] for row in cursor.fetchall()]
            conn.close()
            return threads
        except Exception as e:
            print(f"⚠️ 获取线程列表失败: {e}")
            return []
    
    def cleanup_old_data(self, days_old: int = 30):
        """
        清理旧数据
        
        Args:
            days_old: 删除多少天前的数据
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 删除旧的检查点
            cursor.execute("""
                DELETE FROM checkpoints 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days_old))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            print(f"🗑️ 已清理 {deleted_count} 个旧检查点")
        except Exception as e:
            print(f"⚠️ 清理数据失败: {e}")

# 4. 创建带记忆的图节点
class MemoryAwareNodes:
    """带记忆功能的图节点"""
    
    def __init__(self, memory_manager: LocalMemoryManager):
        self.memory_manager = memory_manager
    
    def analyze_message(self, state: ConversationState, config: RunnableConfig) -> Dict[str, Any]:
        """
        分析消息并保存重要信息到记忆
        
        Args:
            state: 对话状态
            config: 配置信息
        
        Returns:
            Dict: 更新的状态
        """
        print(f"🔍 分析消息...")
        
        # 获取用户ID和最新消息
        user_id = state.get("user_id", "unknown")
        messages = state.get("messages", [])
        
        if not messages:
            return {"messages": []}
        
        last_message = messages[-1]
        
        # 简单的记忆提取逻辑
        if isinstance(last_message, dict):
            content = last_message.get("content", "")
        else:
            content = str(last_message)
        
        # 检测用户偏好
        if "喜欢" in content or "爱" in content:
            preferences = {"preference": content, "detected_at": datetime.now().isoformat()}
            self.memory_manager.save_user_memory(user_id, "preferences", preferences)
        
        # 检测事实信息
        if "我是" in content or "我的名字" in content:
            facts = {"fact": content, "detected_at": datetime.now().isoformat()}
            self.memory_manager.save_user_memory(user_id, "facts", facts)
        
        # 返回原始消息
        return {"messages": messages}
    
    def generate_response(self, state: ConversationState, config: RunnableConfig) -> Dict[str, Any]:
        """
        生成回复，结合历史记忆
        
        Args:
            state: 对话状态
            config: 配置信息
        
        Returns:
            Dict: 包含AI回复的状态更新
        """
        print(f"🤖 生成回复...")
        
        user_id = state.get("user_id", "unknown")
        messages = state.get("messages", [])
        
        # 获取用户记忆
        preferences = self.memory_manager.get_user_memories(user_id, "preferences")
        facts = self.memory_manager.get_user_memories(user_id, "facts")
        
        # 获取最新消息
        if messages:
            if isinstance(messages[-1], dict):
                last_content = messages[-1].get("content", "")
            else:
                last_content = str(messages[-1])
        else:
            last_content = ""
        
        # 构建回复
        response_parts = []
        
        # 基础回复
        if "你好" in last_content or "hi" in last_content.lower():
            response_parts.append("你好！")
            
            # 如果有记忆，个性化问候
            if facts:
                latest_fact = facts[-1]
                if "名字" in latest_fact.get("fact", ""):
                    response_parts.append("很高兴再次见到你！")
        
        elif "我的偏好" in last_content or "我喜欢什么" in last_content:
            if preferences:
                response_parts.append("根据我的记忆，你的偏好包括：")
                for pref in preferences[-3:]:  # 显示最近3个偏好
                    response_parts.append(f"- {pref.get('preference', '')}")
            else:
                response_parts.append("我还没有记录你的偏好信息。")
        
        elif "我是谁" in last_content or "关于我" in last_content:
            if facts:
                response_parts.append("根据我的记忆，关于你的信息：")
                for fact in facts[-3:]:  # 显示最近3个事实
                    response_parts.append(f"- {fact.get('fact', '')}")
            else:
                response_parts.append("请告诉我更多关于你的信息。")
        
        else:
            response_parts.append(f"我收到了你的消息：{last_content}")
        
        # 组合回复
        response_text = "\n".join(response_parts)
        
        # 创建AI消息
        ai_message = {
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        }
        
        return {"messages": [ai_message]}

# 5. 创建完整的记忆图
def create_memory_graph(memory_manager: LocalMemoryManager):
    """
    创建带记忆功能的图
    
    Args:
        memory_manager: 内存管理器
    
    Returns:
        编译后的图
    """
    print("🔨 创建记忆图...")
    
    # 创建节点
    nodes = MemoryAwareNodes(memory_manager)
    
    # 创建状态图
    workflow = StateGraph(ConversationState)
    
    # 添加节点
    workflow.add_node("analyze", nodes.analyze_message)
    workflow.add_node("respond", nodes.generate_response)
    
    # 添加边
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "respond")
    workflow.add_edge("respond", END)
    
    # 编译图，加入检查点和存储
    graph = workflow.compile(
        checkpointer=memory_manager.checkpointer,
        store=memory_manager.store
    )
    
    print("✅ 记忆图创建完成")
    return graph

# 6. 示例使用
def main():
    """主函数 - 演示本地内存保存功能"""
    print("🚀 LangGraph本地内存保存示例")
    print("=" * 50)
    
    # 创建内存管理器
    memory_manager = LocalMemoryManager("conversation_memory.db")
    
    # 创建图
    graph = create_memory_graph(memory_manager)
    
    # 模拟多轮对话
    user_id = "user_123"
    thread_id = "conversation_001"
    
    conversations = [
        "你好！我是小明",
        "我喜欢吃披萨",
        "我也爱看电影",
        "你还记得我喜欢什么吗？",
        "我是谁？"
    ]
    
    print(f"\n💬 开始对话 (用户: {user_id}, 线程: {thread_id})")
    print("-" * 50)
    
    for i, user_message in enumerate(conversations, 1):
        print(f"\n回合 {i}:")
        print(f"👤 用户: {user_message}")
        
        # 创建配置
        config = {"configurable": {"thread_id": thread_id}}
        
        # 创建初始状态
        initial_state = {
            "messages": [{"role": "user", "content": user_message}],
            "user_id": user_id,
            "conversation_context": {}
        }
        
        # 执行图
        try:
            result = graph.invoke(initial_state, config)
            
            # 输出AI回复
            ai_messages = [msg for msg in result["messages"] if msg.get("role") == "assistant"]
            if ai_messages:
                print(f"🤖 AI: {ai_messages[-1]['content']}")
            
        except Exception as e:
            print(f"❌ 执行错误: {e}")
    
    # 演示记忆功能
    print(f"\n📚 记忆功能演示:")
    print("-" * 30)
    
    # 显示用户偏好
    preferences = memory_manager.get_user_memories(user_id, "preferences")
    print(f"用户偏好 ({len(preferences)} 条):")
    for pref in preferences:
        print(f"  - {pref.get('preference', '')}")
    
    # 显示用户事实
    facts = memory_manager.get_user_memories(user_id, "facts")
    print(f"用户事实 ({len(facts)} 条):")
    for fact in facts:
        print(f"  - {fact.get('fact', '')}")
    
    # 显示所有线程
    threads = memory_manager.list_all_threads()
    print(f"所有对话线程 ({len(threads)} 个): {threads}")
    
    # 新线程测试
    print(f"\n🔄 新线程测试:")
    print("-" * 30)
    
    new_thread_id = "conversation_002"
    config2 = {"configurable": {"thread_id": new_thread_id}}
    
    # 在新线程中询问偏好 (应该能访问跨线程记忆)
    test_state = {
        "messages": [{"role": "user", "content": "你还记得我的偏好吗？"}],
        "user_id": user_id,  # 同一用户
        "conversation_context": {}
    }
    
    try:
        result = graph.invoke(test_state, config2)
        ai_messages = [msg for msg in result["messages"] if msg.get("role") == "assistant"]
        if ai_messages:
            print(f"🤖 AI (新线程): {ai_messages[-1]['content']}")
    except Exception as e:
        print(f"❌ 新线程测试错误: {e}")
    
    print(f"\n📊 总结:")
    print("✅ 线程级记忆: 每个对话线程保持独立的对话历史")
    print("✅ 跨线程记忆: 用户偏好和事实在所有线程间共享")
    print("✅ 本地持久化: 数据保存在SQLite数据库中")
    print("✅ 会话恢复: 可以在任何时候恢复之前的对话")
    
    # 清理演示
    response = input("\n❓ 是否清理演示数据? (y/n): ")
    if response.lower() == 'y':
        memory_manager.cleanup_old_data(0)  # 清理所有数据
        print("🗑️ 演示数据已清理")

if __name__ == "__main__":
    main() 