'''
🤪🤪🤪Author: JY
Date: 2026-04-27 12:16:52
LastEditTime: 2026-04-28 13:41:20
'''
import os
import json
import hashlib
from typing import Tuple, List
from .logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def get_file_md5_hex(file_path: str) -> str: #获取文件的md5的十六进制字符串：去重
    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件{file_path}不存在。")
        return
    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]文件{file_path}不是文件。")
        return
    md5_obj = hashlib.md5()
 
    chunk_size = 4096  # 4KB分片，避免文件过大爆内存
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
        '''
        相当于:
        chunk = f.read(chunk_size)
        while chunk:
            md5_obj.update(chunk)
            chunk = f.read(chunk_size)
        '''
        md5_hex = md5_obj.hexdigest()
        return md5_hex
    except Exception as e:
        logger.error(f"[md5计算]文件{file_path}md5计算失败：{str(e)}")
        return None

                
def listdir_with_allowed_type(folder_path: str, allowed_types: tuple[str]): #返回文件夹内的文件列表(允许的文件后缀)
    files = []

    if not os.path.exists(folder_path):
        logger.error(f"[文件列表]文件夹{folder_path}不存在。")
        return allowed_types 
    
    for f in os.listdir(folder_path):
        if f.endswith(allowed_types):
            files.append(os.path.join(folder_path, f))
    return tuple(files)



def pdf_loader(file_path: str,password: str = None) ->list[Document]:
     #加载pdf文件
    return PyPDFLoader(file_path, password=password).load()

def txt_loader(file_path: str) ->list[Document]:     #加载txt文件
    return TextLoader(file_path, encoding="utf-8").load()

def parse_multimodal_txt(file_path: str) -> Tuple[str, List[str]]:
    """
    解析多模态 TXT 文件（JSON 格式: [带<PIC>的文本, [图片名称列表]]）
    :param file_path: TXT 文件路径
    :return: (文本内容, 图片名称列表)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
        if isinstance(data, list) and len(data) == 2:
            return data[0], data[1]
    except (json.JSONDecodeError, IndexError):
        pass
    # 不是 JSON 格式，按纯文本处理
    return raw, []


def clean_manual_text(text: str) -> str:
    """
    清洗手册文本，去除格式噪声，保留语义完整。
    - 去除冗余重复的章节标记（如多次出现的 # 重要安全说明）
    - 保留有实际意义的标题
    - 规范化空白字符
    - 标准化换行
    """
    import re

    # 1. 去除冗余的章节标记行：单独成行的 # 重要安全说明、# 警告、# 注意、# 危险
    #    （但保留后面跟有具体内容的标题）
    text = re.sub(
        r'(?:^|\n)\s*#\s*(?:重要安全说明|警告|注意|危险)\s*(?=\n|$)',
        '\n',
        text
    )

    # 2. 去除行内多余的 # 标记（如 "# 1. 连接电源时  # 应使用专用插座。" → "# 1. 连接电源时 应使用专用插座。"）
    text = re.sub(r'#(?=\s+\S)', '', text)

    # 3. 将 • 统一为普通连字符
    text = text.replace('•', '-')

    # 4. 将连续的多个换行压缩为单个换行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 5. 去除行首尾的空白
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # 6. 将行内连续的多个空格压缩为单个空格
    text = re.sub(r'[ \t]+', ' ', text)

    # 7. 去除 <PIC> 前后的多余空格
    text = re.sub(r'\s*<PIC>\s*', '<PIC>', text)

    # 8. 恢复 <PIC> 前后的换行（方便后续分块）
    text = re.sub(r'(<PIC>)', r'\n\1\n', text)

    # 9. 再次压缩多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()
