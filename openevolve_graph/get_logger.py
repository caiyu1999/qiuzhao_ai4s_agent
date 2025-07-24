import logging
import os

def get_logger(log_dir: str, log_name: str = "openevolve.log", log_level: str = "INFO"):
    """
    获取一个配置好的logger。
    :param log_dir: 日志文件存放目录。如果为None，则只输出到控制台。
    :param log_name: 日志文件名。
    :param log_level: 日志级别，字符串类型，如"INFO"、"DEBUG"等。
    :return: logging.Logger对象
    """
    logger = logging.getLogger(log_name)
    if logger.hasHandlers():
        return logger  # 防止重复添加handler

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, log_name)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False  # 防止日志重复输出
    return logger
