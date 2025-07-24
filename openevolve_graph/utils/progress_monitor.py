import asyncio
import json
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import websockets
import logging

class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class NodeProgress:
    node_id: str
    node_name: str
    island_id: Optional[str] = None
    status: NodeStatus = NodeStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress_percentage: float = 0.0
    message: str = ""
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self):
        data = asdict(self)
        data['status'] = self.status.value
        return data

class ProgressMonitor:
    def __init__(self, websocket_port: int = 8765):
        self.websocket_port = websocket_port
        self.nodes_progress: Dict[str, NodeProgress] = {}
        self.connected_clients = set()
        self.server = None
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._loop = None
        self._server_thread = None
        
    async def register_client(self, websocket):
        """注册新的WebSocket客户端"""
        self.connected_clients.add(websocket)
        self.logger.info(f"新客户端连接: {websocket.remote_address}")
        
        # 发送当前所有节点状态
        await self.send_all_progress(websocket)
        
        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.remove(websocket)
            self.logger.info(f"客户端断开连接: {websocket.remote_address}")
    
    async def send_all_progress(self, websocket):
        """向单个客户端发送所有进度信息"""
        try:
            progress_data = {
                "type": "all_progress",
                "data": [progress.to_dict() for progress in self.nodes_progress.values()]
            }
            await websocket.send(json.dumps(progress_data, ensure_ascii=False))
        except Exception as e:
            self.logger.error(f"发送进度信息失败: {e}")
    
    async def broadcast_progress(self, node_progress: NodeProgress):
        """向所有连接的客户端广播进度更新"""
        if not self.connected_clients:
            return
            
        message = {
            "type": "progress_update",
            "data": node_progress.to_dict()
        }
        
        # 创建要断开连接的客户端列表
        disconnected_clients = []
        
        for client in self.connected_clients.copy():
            try:
                await client.send(json.dumps(message, ensure_ascii=False))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)
            except Exception as e:
                self.logger.error(f"发送消息到客户端失败: {e}")
                disconnected_clients.append(client)
        
        # 移除断开连接的客户端
        for client in disconnected_clients:
            self.connected_clients.discard(client)
    
    def _schedule_broadcast(self, node_progress: NodeProgress):
        """在事件循环中安排广播任务"""
        if self._loop and not self._loop.is_closed():
            try:
                # 使用 call_soon_threadsafe 来从其他线程安全地调度协程
                future = asyncio.run_coroutine_threadsafe(
                    self.broadcast_progress(node_progress), 
                    self._loop
                )
                # 不等待结果，让它在后台运行
            except Exception as e:
                self.logger.error(f"调度广播任务失败: {e}")
    
    def start_node(self, node_id: str, node_name: str, island_id: Optional[str] = None, message: str = ""):
        """标记节点开始执行"""
        with self._lock:
            progress = NodeProgress(
                node_id=node_id,
                node_name=node_name,
                island_id=island_id,
                status=NodeStatus.RUNNING,
                start_time=time.time(),
                message=message
            )
            self.nodes_progress[node_id] = progress
            
        # 安排异步广播更新
        self._schedule_broadcast(progress)
        self.logger.info(f"节点开始执行: {node_name} (ID: {node_id})")
    
    def update_node_progress(self, node_id: str, progress_percentage: float, message: str = "", details: Optional[Dict[str, Any]] = None):
        """更新节点执行进度"""
        with self._lock:
            if node_id in self.nodes_progress:
                progress = self.nodes_progress[node_id]
                progress.progress_percentage = progress_percentage
                progress.message = message
                if details:
                    progress.details = details
                
                # 安排异步广播更新
                self._schedule_broadcast(progress)
                self.logger.info(f"节点进度更新: {progress.node_name} - {progress_percentage}%")
    
    def complete_node(self, node_id: str, message: str = "执行完成", details: Optional[Dict[str, Any]] = None):
        """标记节点执行完成"""
        with self._lock:
            if node_id in self.nodes_progress:
                progress = self.nodes_progress[node_id]
                progress.status = NodeStatus.COMPLETED
                progress.end_time = time.time()
                progress.progress_percentage = 100.0
                progress.message = message
                if details:
                    progress.details = details
                
                # 安排异步广播更新
                self._schedule_broadcast(progress)
                self.logger.info(f"节点执行完成: {progress.node_name} (ID: {node_id})")
    
    def fail_node(self, node_id: str, error_message: str, details: Optional[Dict[str, Any]] = None):
        """标记节点执行失败"""
        with self._lock:
            if node_id in self.nodes_progress:
                progress = self.nodes_progress[node_id]
                progress.status = NodeStatus.FAILED
                progress.end_time = time.time()
                progress.message = error_message
                if details:
                    progress.details = details
                
                # 安排异步广播更新
                self._schedule_broadcast(progress)
                self.logger.error(f"节点执行失败: {progress.node_name} (ID: {node_id}) - {error_message}")
    
    async def start_server(self):
        """启动WebSocket服务器"""
        try:
            self.server = await websockets.serve(
                self.register_client, 
                "localhost", 
                self.websocket_port
            )
            self.logger.info(f"进度监控WebSocket服务器启动在端口 {self.websocket_port}")
            await self.server.wait_closed()
        except Exception as e:
            self.logger.error(f"WebSocket服务器启动失败: {e}")
    
    def start_server_in_thread(self):
        """在新线程中启动WebSocket服务器"""
        def run_server():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self.start_server())
            except Exception as e:
                self.logger.error(f"服务器运行错误: {e}")
            finally:
                self._loop.close()
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        self.logger.info("WebSocket服务器线程已启动")
        
        # 等待一小段时间确保事件循环启动
        time.sleep(0.1)
    
    def get_node_progress(self, node_id: str) -> Optional[NodeProgress]:
        """获取特定节点的进度信息"""
        with self._lock:
            return self.nodes_progress.get(node_id)
    
    def get_all_progress(self) -> List[NodeProgress]:
        """获取所有节点的进度信息"""
        with self._lock:
            return list(self.nodes_progress.values())
    
    def clear_progress(self):
        """清空所有进度信息"""
        with self._lock:
            self.nodes_progress.clear()
        self.logger.info("已清空所有进度信息")

# 全局进度监控实例
progress_monitor = ProgressMonitor(websocket_port=8766) 