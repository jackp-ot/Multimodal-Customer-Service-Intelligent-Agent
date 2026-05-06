'''
重新加载知识库：解析 data/ 下的 TXT/PDF 文件，存入 Milvus 向量数据库
（包含图片名称元数据，用于多模态检索）

注意：会清除旧的 MD5 记录以强制重新加载所有文档
'''
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from rag.vector_store import VectorStoreServer
from utils.config_handler import milvus_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def reload_knowledge_base():
    logger.info("=" * 50)
    logger.info("开始重新加载知识库...")
    logger.info("=" * 50)

    # 清除旧的 MD5 记录，强制重新加载
    md5_path = get_abs_path(milvus_config["md5_hex_store"])
    if os.path.exists(md5_path):
        os.remove(md5_path)
        logger.info(f"已清除 MD5 记录: {md5_path}")

    vs = VectorStoreServer()
    vs.load_documents()

    logger.info("=" * 50)
    logger.info("知识库加载完成！")
    logger.info("=" * 50)


if __name__ == "__main__":
    reload_knowledge_base()
