'''
🤪🤪🤪Author: JY
Date: 2026-04-28 13:50:09
LastEditTime: 2026-04-28 15:19:01
'''
from .config_handler import prompt_config
from .path_tool import get_abs_path
from .logger_handler import logger

def _load_prompt(config_key: str, error_msg: str) -> str:
    """通用提示词加载函数"""
    try:
        prompt_path = get_abs_path(prompt_config[config_key])
    except KeyError as e:
        logger.error(f"[_load_prompt]在yaml配置文件中缺少{config_key}配置项")
        raise e
    
    try:
        return open(prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[_load_prompt]{error_msg}：{str(e)}")
        raise e

def load_system_prompt() -> str:
    return _load_prompt('main_prompt_path', '加载系统提示失败')

def load_rag_prompt() -> str:
    return _load_prompt('rag_summary_prompt_path', '加载rag总结提示词失败')

def load_report_prompt() -> str:
    return _load_prompt('report_prompt_path', '加载报告生成提示词失败')

if __name__ == '__main__':
    print(load_system_prompt())
