
from typing import List, Dict, Any
import re

from langchain.prompts import PromptTemplate, BaseChatPromptTemplate
from langchain.agents import Tool, AgentOutputParser
from langchain.schema import (
    HumanMessage, BaseMessage,
    AgentAction, AgentFinish
)

class ComputerUser(object):
    '''家庭管家角色模板'''

    computerUser_template = """Complete the objective as best you can. You have access to the following tools:

{agent_tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

These were previous tasks you completed:



Begin!

Question: {input}
{agent_scratchpad}"""
    prompt: BaseChatPromptTemplate
    output_parser: AgentOutputParser

    class UserPromptTemplate(BaseChatPromptTemplate):
        template: str

        tools: List[Tool]

        def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
            '''格式花输入，并将输入的消息添加上工具信息'''

            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            formatted = self.template.format(**kwargs)
            return [HumanMessage(content=formatted)]
    
    class CustomOutputParser(AgentOutputParser):
        '''自定义输出解析器'''

        def parse(self, llm_output: str) -> AgentAction | AgentFinish:
            if 'Final Answer:' in llm_output:
                return AgentFinish(return_values={'output': llm_output.split('Final Answer:')[-1].strip()}, log=llm_output)
            
            # Parse out the action and action input
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


    def __init__(self,template = "",user_tools: List[Any] = [], input_variables=['input','intermediate_steps']) -> None:
        if template:
            self.computerUser_template = template

        self.prompt = self.UserPromptTemplate(template=self.computerUser_template, tools=user_tools, input_variables=input_variables,
                                              output_parser= self.CustomOutputParser())
        self.output_parser = self.CustomOutputParser()


