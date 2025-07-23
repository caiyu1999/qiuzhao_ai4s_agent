import logging
import os

def setup_logger(log_dir: str, log_file: str, level=logging.INFO):
    """
    Set up the root logger. This clears any existing handlers.
    """
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # File handler
    file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(file_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    return logger 