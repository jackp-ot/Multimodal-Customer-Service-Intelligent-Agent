'''
🤪🤪🤪Author: JY
Date: 2026-04-28 18:41:41
LastEditTime: 2026-04-29 09:18:10
'''
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from utils.config_handler import rag_config
from langchain_community.embeddings import DashScopeEmbeddings


class BaseModelFactory(ABC):
    @abstractmethod
    def generate(self) -> Optional[Embeddings | ChatOpenAI]:
        pass

class ChatModelFactory(BaseModelFactory):

    def generate(self) -> Optional[ChatOpenAI]:
        return ChatOpenAI(
            model = rag_config["chat_model_name"],
            api_key = rag_config["api_key"],
            base_url = rag_config["base_url"],
        )

class MultimodalChatModelFactory(BaseModelFactory):
    """多模态聊天模型工厂（支持图片输入）"""

    def generate(self) -> Optional[ChatOpenAI]:
        model_name = rag_config.get("vision_model_name", "qwen-vl-plus")
        return ChatOpenAI(
            model=model_name,
            api_key=rag_config["api_key"],
            base_url=rag_config["base_url"],
            max_tokens=4096,
        )

class EmbeddingModelFactory(BaseModelFactory):

    def generate(self) -> Optional[Embeddings]:
        return DashScopeEmbeddings(
            model = rag_config["embedding_model_name"],
            dashscope_api_key = rag_config["api_key"],
        )

chat_model = ChatModelFactory().generate()
multimodal_chat_model = MultimodalChatModelFactory().generate()
embedding_model = EmbeddingModelFactory().generate()
