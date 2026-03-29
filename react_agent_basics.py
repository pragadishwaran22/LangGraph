from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time"""
    return datetime.datetime.now().strftime(format)

tools = [search_tool, get_system_time]

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

response = agent.invoke("When was SpaceX's last launch?")
print(response)