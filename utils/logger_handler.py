'''
🤪🤪🤪Author: JY
Date: 2026-04-27 11:08:20
LastEditTime: 2026-04-27 11:41:44
'''
import logging
from .path_tool import get_abs_path
import os
from datetime import datetime
'''
日志保存的根目录
'''
LOG_ROOT = get_abs_path("logs")
#确保日志目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

#日志格式配置
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
DEFAULT_FORMATTER = logging.Formatter(LOG_FORMAT)

def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file = None, 
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))
    #避免重复添加handle
    if logger.handlers:
        return logger
    #控制台handle
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_FORMATTER)
    logger.addHandler(console_handler)

    #文件handle
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_FORMATTER)
    logger.addHandler(file_handler)
    return logger

#快捷获取日志器
logger = get_logger()

if __name__ == "__main__":
    logger.info("这是一条info日志")
    logger.debug("这是一条debug日志")
    logger.error("这是一条error日志")
    logger.warning("这是一条warning日志")
    logger.critical("这是一条critical日志")