import os 
import logging
import time
from openevolve_graph.Config import Config

logger = logging.getLogger(__name__)


def _setup_logging(config:Config) -> None:
        """配置日志记录器，支持输出到文件和控制台。"""
        log_dir = config.log_dir or os.path.join(config.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # 获取根日志记录器并设置级别
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.log_level))

        # 创建文件处理器
        log_file = os.path.join(log_dir, f"openevolve_{time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(console_handler)

        logger.info(f"日志将记录到: {log_file}")
        
if __name__ == "__main__":
    config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
    _setup_logging(config)
    logger.info("测试日志")