'''
Milvus 向量数据库操作类
'''
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List, Optional
from utils.config_handler import milvus_config


class MilvusDB(VectorStore):
    def __init__(self, embedding: Embeddings, collection_name=None):
        self.embedding = embedding
        self.collection_name = collection_name or milvus_config["collection_name"]
        self._connect()
        self._init_collection()
    
    def _connect(self):
        """连接 Milvus 服务器"""
        # connections.connect(
        #     alias="default",
        #     host="localhost",
        #     port="19530",
        #     root_user="root",
        #     root_password="Milvus@123"
        # )
        connections.connect(
            alias=milvus_config["alias"],
            host=milvus_config["host"],
            port=milvus_config["port"],
            token=milvus_config["token"]
        )
    
    def _init_collection(self):
        """创建或加载集合"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="create_time", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="operator", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
            ]
            schema = CollectionSchema(fields, description="RAG")
            self.collection = Collection(self.collection_name, schema)
            
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 128}
            }
            self.collection.create_index("embedding", index_params)
        
        self.collection.load()
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        operator: str = "jackpot",
        **kwargs
    ) -> List[str]:
        """
        添加文本到向量库
        :param texts: 文本列表
        :param metadatas: 元数据列表，每个元素是一个字典，可以包含 source 等信息
        :param operator: 操作人
        :return: 插入的 ID 列表
        """
        embeddings = self.embedding.embed_documents(texts)
        
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        sources = [m.get("source", "") if m else "" for m in (metadatas or [{}] * len(texts))]
        
        entities = [
            texts,
            sources,
            [now] * len(texts),
            [operator] * len(texts),
            embeddings
        ]
        
        result = self.collection.insert(entities)
        self.collection.flush()
        
        return [str(id) for id in result.primary_keys]
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_expr: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """
        相似度搜索
        :param query: 查询文本
        :param k: 返回结果数量
        :param filter_expr: 过滤条件
        :return: Document 对象列表
        """
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(embedding, k=k, filter_expr=filter_expr, **kwargs)
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 5,
        filter_expr: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """
        向量搜索
        :param embedding: 查询向量
        :param k: 返回结果数量
        :param filter_expr: 过滤条件
        :return: Document 对象列表
        """
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            filter=filter_expr,
            output_fields=["text", "source", "create_time", "operator"]
        )
        
        documents = []
        for hits in results:
            for hit in hits:
                doc = Document(
                    page_content=hit.entity.get("text", ""),
                    metadata={
                        "source": hit.entity.get("source", ""),
                        "create_time": hit.entity.get("create_time", ""),
                        "operator": hit.entity.get("operator", ""),
                        "score": hit.score
                    }
                )
                documents.append(doc)
        
        return documents
    
    def search(self, query_text, limit=5, filter_expr=None):
        """
        搜索相似内容（保留原有接口兼容性）
        :param query_text: 查询文本
        :param limit: 返回结果数量
        :param filter_expr: 过滤条件
        :return: 搜索结果
        """
        return self.similarity_search(query_text, k=limit, filter_expr=filter_expr)
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: Optional[str] = None,
        **kwargs
    ) -> "MilvusDB":
        """
        从文本列表创建 MilvusDB 实例
        :param texts: 文本列表
        :param embedding: 嵌入函数
        :param metadatas: 元数据列表
        :param collection_name: 集合名称
        :return: MilvusDB 实例
        """
        instance = cls(embedding=embedding, collection_name=collection_name)
        instance.add_texts(texts, metadatas=metadatas, **kwargs)
        return instance
    
    def delete_by_filename(self, filename):
        """
        根据文件名删除数据
        :param filename: 文件名
        :return: 删除结果
        """
        self.collection.delete(f"source == '{filename}'")
        self.collection.flush()
    
    def get_collection(self):
        """获取 Collection 对象，供高级操作使用"""
        return self.collection
