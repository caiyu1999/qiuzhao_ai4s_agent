"""
Socket通信模块 - 支持Program对象序列化
"""

import socket
import threading
import time
import pickle
from typing import Any, Dict
from dataclasses import is_dataclass, asdict
from openevolve_graph.visualization.vis_class import visualize_data, best_program_vis, overall_information_vis
from openevolve_graph.Graph.Graph_state import GraphState
import logging 
logger=logging.getLogger(__name__)




def safe_serialize(data: Any) -> bytes:
    """安全序列化数据为pickle字节流"""
    try:
        return pickle.dumps(data)
    except Exception as e:
        # 如果序列化失败，返回错误信息
        print(f"序列化失败: {str(e)}")
        error_data = {
            "error": f"序列化失败: {str(e)}",
            "data_type": str(type(data)),
            "timestamp": time.time()
        }
        return pickle.dumps(error_data)


class SimpleServer:
    """可视化数据收集服务器"""
    
    def __init__(self, port=8888):
        self.port = port
        self.running = False
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 绑定端口
        self.server_socket.bind(('localhost', self.port))
        self.server_socket.listen(5)
        
        # 可视化数据
        self.vis_data = visualize_data()
        self.data_lock = threading.Lock()  # 线程安全锁
        self.message_count = 0
        
    def init_vis_data(self, state: GraphState):
        """初始化可视化数据"""
        with self.data_lock:
            self.vis_data.update_all(state)
    
    def get_vis_data(self) -> visualize_data:
        """获取当前可视化数据（线程安全）"""
        with self.data_lock:
            return self.vis_data
    
    def start(self):
        """启动服务器"""
        self.running = True
        
        while self.running:
            try:
                # 接受连接
                client_socket, address = self.server_socket.accept()
                
                # 接收数据，循环接收直到完整
                data = b''
                while True:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                
                try:
                    data_dict = pickle.loads(data)
                    self.message_count += 1
                    
                    # 更新可视化数据
                    with self.data_lock:
                        self._process_message(data_dict)
                        
                except Exception as e:
                    print(f"Error decoding or processing message: {e}")
                    pass
                
                # 关闭连接
                client_socket.close()
                
            except Exception as e:
                if self.running:
                    print(f"Server error: {e}")
                    break
    
    def _process_message(self, data_dict: dict):
        """处理收到的消息并更新可视化数据"""
        try:
            self.vis_data.update_after_node(data_dict)

        except Exception as e:
            print(f"Error processing message in SimpleServer: {e}")


    def stop(self):
        """停止服务器"""
        self.running = False
        if hasattr(self, 'server_socket'):
            self.server_socket.close()


class SimpleClient:
    """简单的客户端 - 支持Program对象序列化"""
    
    def __init__(self, port=8888):
        self.port = port
        
    def send_message(self, message: Any):
        """发送消息 - 自动序列化复杂对象"""
        try:
            # 创建socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # 连接服务器
            client_socket.connect(('localhost', self.port))
            
            # 序列化消息
            if isinstance(message, str):
                # 如果是字符串，包装成字典
                msg_data = {"message": message}
            else:
                msg_data = message
            
            # 序列化为pickle字节流
            serialized_data = safe_serialize(msg_data)
            
            # 发送数据，确保完整发送
            client_socket.sendall(serialized_data)
            
            # 关闭连接
            client_socket.close()
            
        except Exception as e:
            pass


# 测试代码
if __name__ == "__main__":
    # 启动服务器
    server = SimpleServer(port=8889)
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()
    
    # 等一下让服务器启动
    time.sleep(1)
    
    # 测试发送消息
    client = SimpleClient(port=8889)
    test_message = {
        "island_id": "test_island",
        "node_name": "evaluate",
        "update_dict": {"status": "completed", "best_program_id": "prog_123", "best_program_metrics": {"accuracy": 0.95}},
        "state_summary": {"iteration": 5, "num_programs": 10, "now_meeting": 2}
    }
    client.send_message(test_message)

    test_best_prog_update = {
        "best_program": {
            "id": "best_prog_ever",
            "code": "print('hello world')",
            "metrics": {"accuracy": 0.99}
        }
    }
    client.send_message(test_best_prog_update)
    
    time.sleep(1)
    server.stop()