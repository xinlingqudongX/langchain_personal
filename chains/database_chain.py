from typing import Any, Dict, List, Optional
from langchain.schema.output import LLMResult
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain, SQLDatabaseSequentialChain
from langchain.chains.llm import LLMChain
from langchain.schema.output_parser import BaseOutputParser
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.prompts.prompt import PromptTemplate

from llm.ChatGLM2 import CustomChatGLM2

class SqliteOutParser(BaseOutputParser):
    
    def parse(self, text: str) -> str:
        print(text,'解析')
        return super().parse(text)

class SqliteCallback(BaseCallbackHandler):

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        print(serialized)
        print('模型输入:',prompts)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print(response)

templateStr = ''

def sqliteChain():
    db = SQLDatabase.from_uri('sqlite:///database/time_spectrum.db')
    llm = CustomChatGLM2(verbose=True, callbacks=[SqliteCallback()])
    prompt = PromptTemplate(template=templateStr, input_variables=[''])
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=prompt)
    # db_chain.use_query_checker = False

    return db_chain

if __name__ == '__main__':
    sqliteChain()