import logging
import os
import time

def get_logger(log_dir: str, log_name: str = "openevolve.log", log_level: str = "INFO"):
    """
    获取一个配置好的logger，并确保所有子logger（如 logging.getLogger("openevolve_graph.xxx")）的日志也能输出到同一文件和控制台。
    :param log_dir: 日志文件存放目录。如果为None，则只输出到控制台。
    :param log_name: 日志文件名。
    :param log_level: 日志级别，字符串类型，如"INFO"、"DEBUG"等。
    :return: logging.Logger对象
    """
    # 获取根logger（主logger），所有子logger会继承handler
    logger = logging.getLogger(log_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 检查是否已经添加过handler，防止重复添加
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    # 控制台输出
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    # 文件输出
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, log_name)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 允许子logger将日志传递到父logger（默认 propagate=True）
    # 这样所有子logger（如 logging.getLogger("openevolve_graph.xxx")）的日志也会输出到这里设置的handler
    logger.propagate = True

    return logger

def setup_root_logger(log_dir: str = "", log_level: str = "INFO"):
    """
    设置根logger，确保所有子logger都能正确输出。
    这是推荐的配置方式，因为它会捕获所有logger的输出。
    
    :param log_dir: 日志文件存放目录。如果为None，则只输出到控制台。
    :param log_level: 日志级别，字符串类型，如"INFO"、"DEBUG"等。
    """
    # 获取根logger
    root_logger = logging.getLogger()
    
    # 清空现有的handlers
    root_logger.handlers.clear()
    
    # 设置日志级别
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # 创建formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台输出
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # root_logger.addHandler(console_handler)
    
    # 文件输出
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"openevolve_{time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 确保子logger能传播到根logger（这是默认行为，但明确设置）
    root_logger.propagate = True
    
    return root_logger
