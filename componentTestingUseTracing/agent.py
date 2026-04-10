import os

from deepeval.tracing import observe
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
# Import create_agent for the autonomous loop
from langchain.agents import create_agent 
from langchain.agents.middleware import dynamic_prompt, ModelRequest
import sys

# 1. Get the path to the project root relative to this file
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 2. Add the project root to sys.path so `notebooks` can be imported
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from notebooks.RAG.src import RagRetriever,VectorStoreManager,EmbeddingManager

# --- 1. THE RAG MIDDLEWARE ---
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """This runs automatically EVERY time the agent 'thinks'"""
    # Get the last thing the user said
    last_query = request.state["messages"][-1].content
    

    # Initialize managers
    persist_directory = os.path.join(project_root, "notebooks", "RAG", "store")
    vector_store_manager = VectorStoreManager(persist_directory=persist_directory)
    embeddings_manager = EmbeddingManager()

    # Initialize RAG Retriever
    rag_retriever = RagRetriever(vector_store_manager, embeddings_manager)

    # Example query
   
    docs_content = rag_retriever.retrieve(last_query, top_k=3)

    return (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer. "
        f"\n\nContext:\n{docs_content}"
    )

# --- 2. THE TOOLS ---
@tool
@observe(type='tool')
def search() -> str:
    """Search for information the president of USA."""
    return f"Results for: president of USA is Donald Trump"

@tool
@observe(type='tool')
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

# # --- 4. EXECUTION ---
# # Input must be a dictionary with a "messages" key for create_agent
# response = rag_agent.invoke({
#     "messages": [HumanMessage(content="What is he total experience of Gautam?")]
# })

# # Access the final message in the state
# print(response["messages"][-1].content)


@observe(type='agent',available_tools=['search','get_weather'])
def agent_with_tools():
    # --- 3. THE AGENT SETUP ---
    model = ChatOpenAI(
        base_url="http://127.0.0.1:8080", 
        api_key="dummy"
    )

    # Use create_agent to build the "Robot"
    # Note: We pass the middleware here so the RAG logic is active!
    return create_agent(
        model=model,
        tools=[search, get_weather]
    )
    
