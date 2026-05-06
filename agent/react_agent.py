'''
🤪🤪🤪Author: JY
Date: 2026-04-29 19:18:32
LastEditTime: 2026-04-29 21:44:19
'''
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from model.factory import chat_model
from utils.prompt_load import load_system_prompt
from agent.tools.agent_tools import rag_retrieve_tool
from agent.tools.middle_ware import monitor_tool,log_before_model

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model = chat_model,
            system_prompt = load_system_prompt(),
            tools=[rag_retrieve_tool],
            middleware=[monitor_tool,log_before_model]
        )

    def execute_stream(self,query: str):
        input_dict = {
            "messages": [
                {"role": "user","content": query},
            ]
        }
        for chunk in self.agent.stream(input_dict,stream_mode="values"):
            latest_message = chunk["messages"][-1]
            # 只输出最终的 AI 回答（跳过中间的工具调用、检索结果和思考过程）
            if isinstance(latest_message, AIMessage) and not latest_message.tool_calls:
                if latest_message.content:
                    yield latest_message.content.strip()+"\n"

if __name__ == "__main__":
    react_agent = ReactAgent()
    query = "手表表带尺寸有哪些？"
    for chunk in react_agent.execute_stream(query):
        print(chunk,end="",flush=True)
            
       
