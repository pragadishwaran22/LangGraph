from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
import datetime
import sys

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time"""
    return datetime.datetime.now().strftime(format)

tools = [search_tool, get_system_time]

agent = create_agent(
    model=llm,
    tools=tools,
)

response = agent.invoke({"messages": [("user", "give me a funny tweet about today's weather in chennai")]})
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
print(response["messages"][-1].content)