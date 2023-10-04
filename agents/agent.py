from langchain.callbacks import HumanApprovalCallbackHandler
from langchain.tools import ShellTool

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import ChatGLM