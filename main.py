from uuid import UUID
from langchain.schema.output import LLMResult
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union, Optional, Dict, List, Any, Mapping
from torch.nn import Module
import os
from pathlib import Path
import pyttsx3
from pyttsx3 import Engine
import re
from tempfile import TemporaryDirectory
import platform
from llm.ChatGLM2 import CustomChatGLM2

from langchain.agents import (
    ZeroShotAgent, Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser,
    initialize_agent, AgentType, load_tools, Agent
)

from langchain.agents.react.base import DocstoreExplorer
from langchain.agents.agent_toolkits import FileManagementToolkit, SQLDatabaseToolkit, PlayWrightBrowserToolkit

from langchain.llms.chatglm import ChatGLM
from langchain.llms.base import LLM

from langchain.callbacks.manager import CallbackManager, CallbackManagerForLLMRun
from langchain.requests import TextRequestsWrapper

from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate, BaseChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage, AgentAction, AgentFinish
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.chains import SimpleSequentialChain, SequentialChain, TransformChain
from langchain.chains.llm import LLMChain
from langchain.chains.router import MultiPromptChain, MultiRouteChain
from langchain.tools import BaseTool
from langchain.utilities import PythonREPL, BashProcess, BingSearchAPIWrapper, GoogleSearchAPIWrapper
from langchain.sql_database import SQLDatabase

from chains.database_chain import sqliteChain


PlayEngine = pyttsx3.init()
# 搜索工具
class SearchTool(BaseTool):
    name = "Search"
    description = "如果我想知道天气，'鸡你太美'这两个问题时，请使用它"
    return_direct = True  # 直接返回结果

    def _run(self, query: str) -> str:
        print("\nSearchTool query: " + query)
        return "这个是一个通用的返回"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")

# 计算工具
class CalculatorTool(BaseTool):
    name = "Calculator"
    description = "如果是关于数学计算的问题，请使用它"

    def _run(self, query: str) -> str:
        print("\nCalculatorTool query: " + query)
        return "3"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")

  

class LlmClllback(BaseCallbackHandler):

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        # print(serialized)
        # print('模型输入:',prompts)
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print('模型返回结果:',response)
        pass
    

def main():
    #   执行模型位置
    # tokenizer = AutoTokenizer.from_pretrained(r"d:\project\ChatGLM2-6B\model\6B-int4", trust_remote_code=True)
    # model = AutoModel.from_pretrained(r"d:\project\ChatGLM2-6B\model\6B-int4", trust_remote_code=True).cuda()
    # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
    # model = load_model_on_gpus(r"d:\project\ChatGLM2-6B\model\6B-int4", num_gpus=1)
    # model = model.eval()

    # model.stream_chat(tokenizer, query, history=history,past_key_values=past_key_values,return_past_key_values=True)
    systemType = platform.system().lower()

    custom_prompt = PromptTemplate(template=f'''
你是桌面AI助手，需要用最简洁的方式回答用户的提问，当回答包含有其他语言或者命令时带上相关标识。

当前的已知条件:
当前的系统: {systemType}

用户的提问: {{question}}
''', input_variables=["question"])
    
    # Set up the base template
    # Set up the base template

    llm = CustomChatGLM2(verbose=True, callbacks=[LlmClllback()])

    # tools_names = ["python_repl","requests_all","terminal"]
    
    # agent_tools = load_tools(tools_names, llm=llm)

    # fileKit = FileManagementToolkit(root_dir=str(TemporaryDirectory().name))

    # action_tools = [
    #     Tool(name="Python 运行环境", func=PythonREPL.run, description="用于执行python代码"),
    #     Tool(name="Linux 脚本运行环境", func=BashProcess.run, description="用于Linux系统执行shell脚本的运行环境"),
    #     Tool(name="Bing 搜索引擎", func=BingSearchAPIWrapper.run, description="当我没有使用代理的时候，使用此搜索引擎"),
    #     Tool(name="Google 网络搜索", func=GoogleSearchAPIWrapper.run, description="当我使用代理的时候，使用此搜索引擎"),
    # ]

    #   chain是固定的调用链，类似执行固定的程序
    llmChain = LLMChain(prompt=custom_prompt, llm=llm, verbose=True, callbacks=[LlmClllback()])
    # llmChain = sqliteChain()

    # agent = initialize_agent(tools, llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # agent = LLMSingleActionAgent(llm_chain=llmChain, output_parser=output_paresr, stop=["\nObservation:"])

    # agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=[], verbose=True)

#     physics_template = """You are a very smart physics professor. \
# You are great at answering questions about physics in a concise and easy to understand manner. \
# When you don't know the answer to a question you admit that you don't know.
 
# Here is a question:
# {input}"""
 
#     math_template = """You are a very good mathematician. You are great at answering math questions. \
# You are so good because you are able to break down hard problems into their component parts, \
# answer the component parts, and then put them together to answer the broader question.
 
# Here is a question:
# {input}"""
#     prompt_infos = [
#     {
#         "name": "physics", 
#         "description": "Good for answering questions about physics", 
#         "prompt_template": physics_template
#     },
#     {
#         "name": "math", 
#         "description": "Good for answering math questions", 
#         "prompt_template": math_template
#     }
# ]
    # llmChain = MultiPromptChain.from_prompts(llm, prompt_infos, verbose=True)
 
 
    
    while True:
        query = input("问题:")
        mesages = [
            HumanMessage(content=query)
        ]
        try:
            llm_result = llmChain.run(query)
            print('结果:', llm_result)
            if llm_result and PlayEngine:
                PlayEngine.say(llm_result)
                PlayEngine.runAndWait()
        except ValueError as err:
            print(err)
        # res = chain.run(query)


if __name__ == '__main__':
    main()