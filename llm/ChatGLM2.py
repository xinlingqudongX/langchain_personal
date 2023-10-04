from langchain.llms.base import LLM
from typing import List, Union, Optional, Any
from model import LocalModel

from langchain.callbacks.manager import CallbackManager, CallbackManagerForLLMRun

class CustomChatGLM2(LLM):
    '''自定义ChatGLM2模型'''

    logging: bool = False
    output_keys: List[str] = ['output']
    llm_type:str = 'chatglm'
    
    @property
    def _llm_type(self) -> str:
        return self.llm_type
    
    def log(self, log_str):
        if self.logging:
            print(log_str)
        else:
            return
    
    def _call(self, prompt: str, stop: List[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None) -> str:
        self.log('开始调用')
        self.log(prompt)

        if self._llm_type != "chatglm":
            return '不支持该类型的llm'
        
        past_key_values, history = None, []
        current_length = 0
        anawer_res = ''
        for response, history, past_key_values in LocalModel.instance().chatGLM2.stream_chat(
            LocalModel.instance().chatglm2Tokenizer, prompt, history=history, 
            past_key_values=past_key_values, return_past_key_values=True
        ):
            print(response[current_length:], end="", flush=True)


            current_length = len(response)
            anawer_res = response
        
        return anawer_res
