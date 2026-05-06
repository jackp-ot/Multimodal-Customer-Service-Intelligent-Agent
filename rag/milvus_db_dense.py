'''
Milvus 向量数据库操作类（支持多模态图片引用）
'''
import os
import json
import pickle
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List, Optional
from utils.config_handler import milvus_config
from utils.logger_handler import logger
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus import AnnSearchRequest, RRFRanker

class MilvusDB(VectorStore):
    def __init__(self, embedding: Embeddings, collection_name=None, bm25_path=None):
        self.embedding = embedding
        self.collection_name = collection_name or milvus_config["collection_name"]
        self.bm25_path = bm25_path or os.path.join(os.path.dirname(__file__), "bm25_model.pkl")
        self.bm25 = None
        self.bm25_fitted = False
        self._connect()
        self._init_collection()
        self._load_bm25_if_exists()

    def _load_bm25_if_exists(self):
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
                self.bm25_fitted = True

    def _save_bm25(self):
        if self.bm25 is not None:
            os.makedirs(os.path.dirname(self.bm25_path), exist_ok=True)
            with open(self.bm25_path, "wb") as f:
                pickle.dump(self.bm25, f)

    def fit_bm25(self, corpus: List[str]):
        self.bm25 = BM25EmbeddingFunction()
        self.bm25.fit(corpus)
        self.bm25_fitted = True
        self._save_bm25()

    def _connect(self):
        connections.connect(
            alias=milvus_config["alias"],
            host=milvus_config["host"],
            port=milvus_config["port"],
            token=milvus_config["token"]
        )

    def _init_collection(self):
        if utility.has_collection(self.collection_name):
            temp_collection = Collection(self.collection_name)
            existing_fields = [f.name for f in temp_collection.schema.fields]

            if "image_names" not in existing_fields:
                # 集合缺少 image_names 字段，删除重建
                logger.warning(f"[Milvus] 集合 {self.collection_name} 缺少 image_names 字段，重建中...")
                temp_collection.release()
                utility.drop_collection(self.collection_name)
                self._create_collection()
            else:
                # 检查 image_names 字段长度是否足够
                img_field = next((f for f in temp_collection.schema.fields if f.name == "image_names"), None)
                current_max = img_field.params.get("max_length", 0) if img_field else 0
                if current_max < 4096:
                    logger.warning(f"[Milvus] image_names 字段长度({current_max})不足，需重建为 4096...")
                    temp_collection.release()
                    utility.drop_collection(self.collection_name)
                    self._create_collection()
                else:
                    self.collection = temp_collection
        else:
            self._create_collection()

        self.collection.load()

    def _create_collection(self):
        """创建 Milvus 集合（含 image_names 字段）"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="image_names", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="create_time", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        schema = CollectionSchema(fields, description="RAG (multi-modal)")
        self.collection = Collection(self.collection_name, schema)

        dense_index = {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 16, "efConstruction": 200}
        }
        self.collection.create_index("dense_embedding", dense_index)

        sparse_index = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP"
        }
        self.collection.create_index("sparse_embedding", sparse_index)

        self.collection.load()

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs
    ) -> List[str]:
        dense_embeddings = self.embedding.embed_documents(texts)
        if self.bm25 is None:
            raise RuntimeError("请先调用 fit_bm25() 训练 BM25 模型")
        sparse_embeddings_matrix = self.bm25.encode_documents(texts)
        sparse_embeddings = [
            {int(idx): float(val) for idx, val in zip(row.indices, row.data)}
            for row in sparse_embeddings_matrix
        ]

        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sources = [m.get("source", "") if m else "" for m in (metadatas or [{}] * len(texts))]
        image_names = [m.get("image_names", "") if m else "" for m in (metadatas or [{}] * len(texts))]

        entities = [
            texts,
            sources,
            image_names,
            [now] * len(texts),
            dense_embeddings,
            sparse_embeddings
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
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(embedding, k=k, filter_expr=filter_expr, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 5,
        filter_expr: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        search_params = {
            "metric_type": "IP",
            "params": {"ef": 64}
        }

        results = self.collection.search(
            data=[embedding],
            anns_field="dense_embedding",
            param=search_params,
            limit=k,
            filter=filter_expr,
            output_fields=["text", "source", "image_names", "create_time"]
        )

        documents = []
        for hits in results:
            for hit in hits:
                doc = Document(
                    page_content=hit.entity.get("text", ""),
                    metadata={
                        "source": hit.entity.get("source", ""),
                        "create_time": hit.entity.get("create_time", ""),
                        "image_names": hit.entity.get("image_names", ""),
                        "score": hit.score
                    }
                )
                documents.append(doc)

        return documents

    def search(self, query_text, limit=5, filter_expr=None):
        return self.similarity_search(query_text, k=limit, filter_expr=filter_expr)

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> List[Document]:
        dense_embedding = self.embedding.embed_query(query)

        if self.bm25 is None:
            raise RuntimeError("请先调用 fit_bm25() 训练 BM25 模型")
        sparse_embedding = self.bm25.encode_queries([query])[0]

        sparse_csr = sparse_embedding.tocsr()
        sparse_dict = {int(idx): float(val) for idx, val in zip(sparse_csr.indices, sparse_csr.data)}

        dense_req = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=k * 10
        )

        sparse_req = AnnSearchRequest(
            data=[sparse_dict],
            anns_field="sparse_embedding",
            param={"metric_type": "IP"},
            limit=k * 10
        )

        results = self.collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=RRFRanker(),
            limit=k,
            output_fields=["text", "source", "image_names", "create_time"]
        )

        documents = []
        for hits in results:
            for hit in hits:
                doc = Document(
                    page_content=hit.entity.get("text", ""),
                    metadata={
                        "source": hit.entity.get("source", ""),
                        "create_time": hit.entity.get("create_time", ""),
                        "image_names": hit.entity.get("image_names", ""),
                        "score": hit.score
                    }
                )
                documents.append(doc)

        return documents

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: Optional[str] = None,
        **kwargs
    ) -> "MilvusDB":
        instance = cls(embedding=embedding, collection_name=collection_name)
        instance.add_texts(texts, metadatas=metadatas, **kwargs)
        return instance

    def delete_by_filename(self, filename):
        self.collection.delete(f"source == '{filename}'")
        self.collection.flush()

    def get_collection(self):
        return self.collection
