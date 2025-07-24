#!/usr/bin/env python3
"""
独立的WebSocket服务器，用于实时传输节点执行进度到前端
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from openevolve_graph.utils.progress_monitor import progress_monitor
from openevolve_graph.utils.log_setup import setup_logger

async def main():
    """启动WebSocket服务器"""
    # 设置日志
    logger_dir = project_root / "logs"
    logger = setup_logger(str(logger_dir), "websocket_server.log")
    
    logger.info("正在启动WebSocket进度监控服务器...")
    
    try:
        # 启动WebSocket服务器
        await progress_monitor.start_server()
    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"服务器运行错误: {e}")
    finally:
        logger.info("WebSocket服务器已关闭")

if __name__ == "__main__":
    print("启动WebSocket进度监控服务器...")
    print("服务器将在 ws://localhost:8765 上运行")
    print("按 Ctrl+C 停止服务器")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n服务器已停止") 