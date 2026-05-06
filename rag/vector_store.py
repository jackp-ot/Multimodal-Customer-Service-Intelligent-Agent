'''
🤪🤪🤪Author: JY
Date: 2026-04-28 16:08:31
LastEditTime: 2026-04-30 12:15:48
'''
import sys
import os
# 获取项目根目录（Agent/）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.config_handler import milvus_config
# from rag.milvus_db import MilvusDB
from rag.milvus_db_dense import MilvusDB
from model.factory import embedding_model
from rag.txt_chunk import TxtSmartChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex, parse_multimodal_txt, clean_manual_text
from utils.logger_handler import logger
from langchain_core.documents import Document

class VectorStoreServer(object):
    def __init__(self):
        '''
        向量数据库服务
        :param embedding: 嵌入函数
        '''
        self.embedding = embedding_model
        self.vector_store = MilvusDB(embedding=self.embedding)
        self.txt_chunker = TxtSmartChunker(chunk_max_length=1000, chunk_overlap=100)
        self.pdf_spliter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )

    def get_retriever(self, search_kwargs: dict = None):
        '''
        获取检索器
        :param search_kwargs: 检索参数，如 {"k": 5}，默认使用配置中的相似度阈值
        :return: 检索器
        '''
        if search_kwargs is None:
            search_kwargs = {"k": milvus_config["k"]}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def load_documents(self, documents: list = None):
        '''
        从数据文件夹内读取数据，转为向量存入向量数据库
        要计算文件的md5做去重
        '''
        def check_md5_hex(md5_for_check:str):
            if not os.path.exists(get_abs_path(milvus_config["md5_hex_store"])):
                open(get_abs_path(milvus_config["md5_hex_store"]), "w", encoding="utf-8").close()
                return False
            with open(get_abs_path(milvus_config["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True
                return False
        
        def save_md5_hex(md5_for_check:str):
            with open(get_abs_path(milvus_config["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")
        
        def get_file_documents(read_path: str):
            '''
            从文件路径读取文档，转为向量存入向量数据库
            :param read_path: 文档文件路径
            :return: (文档列表, 图片名称列表) 的元组列表
            '''
            if read_path.endswith(".txt"):
                # 尝试解析多模态 JSON 格式
                text, image_names = parse_multimodal_txt(read_path)
                if image_names:
                    # JSON 格式：返回文本和图片名
                    return [(Document(page_content=text, metadata={"source": os.path.basename(read_path)}), image_names)]
                else:
                    # 纯文本格式
                    docs = txt_loader(read_path)
                    return [(doc, []) for doc in docs]
            elif read_path.endswith(".pdf"):
                docs = pdf_loader(read_path)
                return [(doc, []) for doc in docs]
            else:
                return []

        allowed_file_path = listdir_with_allowed_type(
            get_abs_path(milvus_config["data_path"]),
            tuple(milvus_config["allowed_file_type"]))
        
        # 第一阶段：收集所有需要处理的新文档和文本
        pending_tasks = []  # 存储 (path, documents)
        all_new_texts = []  # 存储所有新文本用于 BM25 拟合

        for path in allowed_file_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库] {path} 内容已存在，跳过")
                continue
            
            docs = get_file_documents(path)
            if docs:
                pending_tasks.append((path, docs))
                all_new_texts.extend([doc.page_content for doc, _ in docs])

        # 第二阶段：如果有新数据，先拟合 BM25 模型
        if all_new_texts:
            logger.info(f"[加载知识库] 开始拟合 BM25 模型，共 {len(all_new_texts)} 条文本")
            self.vector_store.fit_bm25(all_new_texts)
            logger.info(f"[加载知识库] BM25 模型拟合完成")
        
        # 第三阶段：分片并插入向量库
        for path, docs in pending_tasks:
            try:
                filename = os.path.basename(path)
                all_chunks = []
                
                for doc_tuple in docs:
                    if path.endswith(".txt"):
                        # doc_tuple can be (Document, image_names) or just Document
                        if isinstance(doc_tuple, tuple):
                            doc, image_names = doc_tuple
                        else:
                            doc = doc_tuple
                            image_names = []

                        # 清洗文本：去除格式噪声
                        cleaned_text = clean_manual_text(doc.page_content)

                        if image_names:
                            chunks = self.txt_chunker.create_multimodal_chunks(
                                cleaned_text, image_names, filename
                            )
                        else:
                            chunks = self.txt_chunker.create_chunks(cleaned_text, filename)

                        for chunk in chunks:
                            all_chunks.append(Document(
                                page_content=chunk["text"],
                                metadata=chunk["metadata"]
                            ))
                    else:
                        doc = doc_tuple[0] if isinstance(doc_tuple, tuple) else doc_tuple
                        split_document: list[Document] = self.pdf_spliter.split_documents([doc])
                        all_chunks.extend(split_document)
                
                if not all_chunks:
                    logger.warning(f"[加载知识库] {path} 分片后没有有效内容，跳过")
                    continue
                
                # 存入向量库 (此时 BM25 已拟合，add_texts 会自动生成稀疏向量)
                self.vector_store.add_documents(all_chunks)
                
                # 记录 MD5
                save_md5_hex(get_file_md5_hex(path)) 
                logger.info(f"[加载知识库] {path} 内容已加载")
            except Exception as e:
                logger.error(f"[加载知识库] {path} 加载失败: {str(e)}", exc_info=True)

    def hybrid_search(self, query: str, k: int = 5):
        """
        混合检索（稠密 + 稀疏）
        :param query: 查询文本
        :param k: 返回结果数量
        :return: Document 列表
        """
        return self.vector_store.hybrid_search(query, k=k)
   
if __name__ == "__main__":
    vs = VectorStoreServer()
    vs.load_documents()
    retriever = vs.get_retriever()
    results = retriever.invoke("手表表带尺寸有哪些？")
    print(results)
