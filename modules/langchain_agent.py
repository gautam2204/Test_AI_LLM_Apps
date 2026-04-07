from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from openai import api_key


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = ChatOpenAI(
    base_url="http://127.0.0.1:8080", 
    api_key="dummy"
)
# 2. Bind the tools to the model
# This handles the serialization that was causing your Pydantic error
model_with_tools = agent.bind_tools([search, get_weather])
messages = [HumanMessage(content="Hello, I am Gautam")]
response = agent.invoke(messages)
print(response)