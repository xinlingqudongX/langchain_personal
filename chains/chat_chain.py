from typing import Any, List, Dict, Optional, Union
import platform

from llm.ChatGLM2 import CustomChatGLM2

from langchain.chains.llm import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts.prompt import PromptTemplate

from langchain.schema.output import LLMResult

class LlmClllback(BaseCallbackHandler):

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        # print(serialized)
        # print('模型输入:',prompts)
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print('模型返回结果:',response)
        pass

systemType = platform.system().lower()
custom_prompt = PromptTemplate(template=f'''
你是桌面AI助手，需要用最简洁的方式回答用户的提问，当回答包含有其他语言或者命令时带上相关标识。

当前的已知条件:
当前的系统: {systemType}

用户的提问: {{question}}
''', input_variables=["question"])


def chat_chain():
    llm = CustomChatGLM2(verbose=True)
    llmChain = LLMChain(prompt=custom_prompt, llm=llm, verbose=True, callbacks=[LlmClllback()])

    return llmChain