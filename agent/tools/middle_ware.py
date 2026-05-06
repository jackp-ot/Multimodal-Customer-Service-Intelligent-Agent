'''
🤪🤪🤪Author: JY
Date: 2026-04-29 16:35:25
LastEditTime: 2026-04-29 21:48:47
'''
from typing import Callable
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.runtime import Runtime
from langchain.agents import AgentState
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
    #请求的数据封装
    request,
    #执行的函数本身
    handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command: #完成工具执行的监控
    logger.info(f"[monitor_tool]工具调用: {request.tool_call['name']}")
    logger.info(f"[monitor_tool]工具传入参数: {request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[monitor_tool]工具{request.tool_call['name']}调用成功")

        # if request.tool_call['name'] == 'report_prompt_switch':
        #     report_prompt_switch()
        return result
    except Exception as e:
        logger.error(f"[monitor_tool]工具{request.tool_call['name']}调用异常:,原因{str(e)}")
        raise e

@before_model #完成模型执行前的日志记录
def log_before_model(
    state: AgentState,  #整个智能体中的状态记录
    runtime: Runtime,  #记录了整个执行过程中的上下文信息
):
    logger.info(f"[log_before_model]即将调用模型，带有 {len(state['messages'])} 条消息")
    if state['messages']:  # 先判断列表不为空
        logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    
    return None


# @dynamic_prompt    #动态切换提示词 每一次在生成提示词之前调用该函数
# def report_prompt_switch(request:ModelRequest):
#     is_report = request.runtime.context.get("report", False)
#     if is_report:
#         return "请根据上下文生成报告"
#     else:
#         return "请根据上下文回答问题"
    

