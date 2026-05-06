'''
🤪🤪🤪Author: JY
Date: 2026-04-29 13:49:13
LastEditTime: 2026-04-30 13:31:25
'''
'''
多模态 RAG 总结服务：用户提问 -> 检索文本+图片 -> 多模态模型理解 -> 返回带<PIC>标记的回答
'''
import sys
import os
import base64
import json
from typing import List, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from utils.prompt_load import load_rag_prompt
from rag.vector_store import VectorStoreServer
from model.factory import multimodal_chat_model, chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from utils.path_tool import get_abs_path
from utils.config_handler import milvus_config
from utils.logger_handler import logger

# 插图目录
ILLUSTRATION_DIR = get_abs_path(os.path.join(milvus_config["data_path"], "插图"))
MAX_IMAGES_PER_REQUEST = 5  # 每次请求最多携带的图片数


class RagSummaryService(object):
    def __init__(self):
        self.vector_store = VectorStoreServer()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        # 优先使用多模态模型，否则回退到文本模型
        self.model = multimodal_chat_model if multimodal_chat_model else chat_model
        self.text_model = chat_model
        self.chain = self.prompt_template | self.text_model | StrOutputParser()

    def retrieve_docs(self, query: str) -> List[Document]:
        return self.vector_store.hybrid_search(query)

    def _load_image_base64(self, image_name: str) -> Optional[str]:
        """
        从插图目录加载图片，转为 base64
        :param image_name: 图片名称（不含扩展名）
        :return: base64 字符串，或 None（图片不存在）
        """
        for ext in [".jpg", ".jpeg", ".png"]:
            path = os.path.join(ILLUSTRATION_DIR, image_name + ext)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    img_data = f.read()
                mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
                b64 = base64.b64encode(img_data).decode("utf-8")
                return f"data:{mime};base64,{b64}"
        return None

    def _extract_images_from_docs(self, docs: List[Document]) -> List[str]:
        """
        从检索结果中提取所有图片名称，去重并限制数量
        """
        all_images = []
        seen = set()

        for doc in docs:
            img_names_str = doc.metadata.get("image_names", "")
            if not img_names_str:
                continue
            try:
                names = json.loads(img_names_str)
                for name in names:
                    if name not in seen:
                        seen.add(name)
                        all_images.append(name)
            except (json.JSONDecodeError, TypeError):
                pass

        return all_images[:MAX_IMAGES_PER_REQUEST]

    def _build_multimodal_messages(self, query: str, context: str, image_b64_list: List[str]) -> list:
        """
        构建多模态消息列表：
        1. System: RAG 总结提示词
        2. Human: 用户问题 + 参考资料 + 图片（交替 text 和 image_url 块）
        """
        system_prompt = self.prompt_text + """

## 图片输出规则
- 参考资料中带有 <PIC> 标记，表示该位置有对应的产品插图
- 回答时请在合适的位置使用 <PIC> 标记来展示相关图片
- <PIC> 标记应该放在对应的文字说明之后，表示需要在此处展示一张产品图
- 如果参考资料中有图片，请务必在回答中引用相关图片来辅助说明
- 所有图片已随参考资料附上，请直接根据内容判断在何处插入 <PIC>
"""

        human_parts = []
        # 先说用户问题和文本参考资料
        human_parts.append({
            "type": "text",
            "text": f"用户提问: {query}\n\n参考资料:\n{context}"
        })

        # 附加图片
        if image_b64_list:
            human_parts.append({
                "type": "text",
                "text": f"\n\n以下是参考资料中对应的产品插图（共 {len(image_b64_list)} 张），请在回答中使用 <PIC> 标记来展示它们："
            })
            for i, img_b64 in enumerate(image_b64_list):
                human_parts.append({
                    "type": "image_url",
                    "image_url": {"url": img_b64}
                })

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_parts),
        ]

    def rag_summary(self, query: str) -> str:
        """
        多模态 RAG 总结
        - 检索相关文档片段
        - 加载对应的产品插图
        - 发送给多模态模型理解
        - 返回带 <PIC> 标记的回答
        """
        docs = self.retrieve_docs(query)

        if not docs:
            logger.info(f"[RAG] 未检索到相关文档，使用纯文本模型")
            return self.chain.invoke({"input": query, "context": "未检索到相关参考资料。"})

        # 构建文本上下文
        context = ""
        for i, doc in enumerate(docs):
            context += f"参考资料{i}: {doc.page_content} | 来源: {doc.metadata.get('source', '')}\n"

        # 提取图片
        image_names = self._extract_images_from_docs(docs)
        image_b64_list = []

        for name in image_names:
            b64 = self._load_image_base64(name)
            if b64:
                image_b64_list.append(b64)

        # 如果有图片且多模态模型可用，使用多模态
        if image_b64_list:
            try:
                logger.info(f"[RAG] 多模态检索: {len(image_b64_list)} 张图片, {len(docs)} 个文本片段")
                messages = self._build_multimodal_messages(query, context, image_b64_list)
                result = self.model.invoke(messages)
                return result.content
            except Exception as e:
                logger.error(f"[RAG] 多模态调用失败: {str(e)}，回退到纯文本", exc_info=True)

        # 回退到纯文本模型
        logger.info(f"[RAG] 纯文本检索: {len(docs)} 个文本片段")
        return self.chain.invoke({"input": query, "context": context})

    def rag_retrieve(self, query: str) -> str:
        """
        仅检索+重排，不经过 LLM 总结
        - 检索相关文档片段（已在 Milvus 层经过 RRF 重排）
        - 直接返回格式化的原始检索结果给 agent 自行处理
        """
        docs = self.retrieve_docs(query)

        if not docs:
            logger.info(f"[RAG] 未检索到相关文档")
            return "未检索到相关参考资料。"

        # 构建纯文本上下文（不做 LLM 总结）
        context = ""
        for i, doc in enumerate(docs):
            # 提取该文档关联的图片名（供模型输出 <PIC> 用）
            img_names = doc.metadata.get("image_names", "")
            if img_names:
                # 校验：文档中的 <PIC> 数与图片名数是否匹配
                pic_count = doc.page_content.count("<PIC>")
                try:
                    img_list = json.loads(img_names)
                    if pic_count != len(img_list):
                        logger.warning(
                            f"[RAG校验] 文档{i+1}中 <PIC> 数({pic_count}) "
                            f"!= 图片名数({len(img_list)}) "
                            f"来源: {doc.metadata.get('source', '')}"
                        )
                except (json.JSONDecodeError, TypeError):
                    pass
                context += f"[{i+1}] {doc.page_content}\n   图片: {img_names}\n"
            else:
                context += f"[{i+1}] {doc.page_content}\n"

        logger.info(f"[RAG] 检索到 {len(docs)} 个文档片段，已跳过 LLM 总结，直传 agent")
        return context.strip()


if __name__ == "__main__":
    rag_service = RagSummaryService()
    summary = rag_service.rag_summary("吹风机有哪些安全注意事项？")
    print(summary)
