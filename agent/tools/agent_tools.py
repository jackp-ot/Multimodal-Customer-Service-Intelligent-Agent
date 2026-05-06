'''
🤪🤪🤪Author: JY
Date: 2026-04-29 15:49:55
LastEditTime: 2026-04-29 21:41:10
'''
from langchain_core.tools import Tool
from rag.rag_service import RagSummaryService

rag = RagSummaryService()

def rag_summary_tool(query: str) -> str:
    """
    搜索工具，用于搜索互联网上的信息
    :param query: 搜索查询
    :return: 搜索结果
    """
    return rag.rag_summary(query)

def rag_retrieve_tool(query: str) -> str:
    """
    检索工具：从知识库检索相关文档片段（已重排），不经过 LLM 总结，返回原始参考资料
    :param query: 搜索查询
    :return: 原始检索结果
    """
    return rag.rag_retrieve(query)

rag_summary_tool = Tool(
    name="rag_summary",
    func=rag_summary_tool,
    description="从向量存储中检索参考资料，根据用户问题搜索参考资料并总结回复"
)

rag_retrieve_tool = Tool(
    name="rag_retrieve",
    func=rag_retrieve_tool,
    description="从向量存储中检索相关文档片段（已重排），直接返回原始参考资料，不做 LLM 总结，适合让 AI 自行处理检索结果"
)