"""
Core Langgraph agent logic for the Personalized Storyteller.

This module defines the agent's state, tools, nodes, edges, and graph
for managing long-term memory and generating personalized stories using
an open-source LLM via Ollama.
"""

import os
import uuid
import faiss
from typing import List, TypedDict, Annotated, Sequence

from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults # Example tool

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# --- Configuration --- 
# Ensure Ollama is running and the desired model (e.g., 'llama3') is pulled.
# Set TAVILY_API_KEY environment variable if using Tavily search.
# os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"

# --- Memory Setup --- 

# Using Ollama embeddings (ensure the embedding model is running in Ollama if needed)
# Or use other embeddings like SentenceTransformers if preferred.
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# FAISS vector store (in-memory)
# Note: For persistence across runs, you'd save/load the index.
# index = faiss.IndexFlatL2(embeddings.client.embed_dim) # Adjust embed_dim based on model
# Using default FAISS setup for simplicity
vector_store = FAISS.from_texts(["Initial dummy entry"], embeddings)

# --- Tool Definitions --- 

def get_user_id(config: RunnableConfig) -> str:
    """Retrieves the user ID from the runnable config."""
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id")
    if user_id is None:
        # In a real app, you might assign a default or raise a more specific error
        print("Warning: User ID not found in config, using default 'guest'")
        return "guest" 
    return user_id

@tool
def save_story_preference(preference: str, config: RunnableConfig) -> str:
    """Save a user's story preference (e.g., genre, character type, setting) to the memory."""
    user_id = get_user_id(config)
    doc_id = str(uuid.uuid4())
    document = Document(
        page_content=preference,
        metadata={"user_id": user_id, "type": "preference", "doc_id": doc_id}
    )
    vector_store.add_documents([document])
    return f"Preference saved: {preference}"

@tool
def search_story_preferences(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant user preferences based on the current conversation or query."""
    user_id = get_user_id(config)
    # Filter search to the specific user_id
    results = vector_store.similarity_search(
        query, k=3, filter={"user_id": user_id}
    )
    if not results:
        return ["No specific preferences found for this user."]
    return [doc.page_content for doc in results]

# Example additional tool (requires TAVILY_API_KEY)
# try:
#     search_tool = TavilySearchResults(max_results=2)
#     tools = [save_story_preference, search_story_preferences, search_tool]
# except ImportError:
#     print("Tavily Search tool not available. Install langchain-community.")
tools = [save_story_preference, search_story_preferences]
tool_node = ToolNode(tools)

# --- Agent State --- 

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    recall_memories: List[str]

# --- Nodes and Edges --- 

# Define the LLM - ensure 'llama3' is available in Ollama
llm = ChatOllama(model="llama3")

# Define the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a creative storyteller agent with long-term memory. "
            "Your goal is to weave engaging narratives based on user requests and preferences. "
            "Use your memory of the user's preferences (genres, characters, settings, past story elements) to personalize the story. "
            "Memory Usage Guidelines:\n"
            "1. Actively use memory tools (`save_story_preference`, `search_story_preferences`) to store and retrieve important details about the user's tastes.\n"
            "2. When asked for a story, use `search_story_preferences` to recall relevant details before generating the narrative.\n"
            "3. If the user expresses a new preference, use `save_story_preference` to remember it.\n"
            "4. Subtly incorporate remembered preferences into the story. Don't just list them.\n"
            "5. If no preferences are found, ask clarifying questions or start with a more general story.\n"
            "6. You can also use other tools like web search if needed to enrich the story (if available).\n\n"
            "## Recall Memories (Context from previous interactions for this user):\n{recall_memories}\n\n"
            "## Instructions\n"
            "Engage with the user naturally. Ask clarifying questions if a request is vague. "
            "When generating a story, make it creative and coherent. "
            "If you need to use a tool, call it. Respond AFTER the tool call is complete."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Agent node function
def agent_node(state: AgentState, config: RunnableConfig):
    """Invokes the LLM to generate a response or decide on tool use."""
    recall_str = "\n".join(state["recall_memories"])
    # Ensure recall_memories is passed correctly
    bound_llm = prompt | llm_with_tools
    response = bound_llm.invoke(
        {"messages": state["messages"], "recall_memories": recall_str},
        config=config
    )
    return {"messages": [response]}

# Memory loading node function
def load_memories_node(state: AgentState, config: RunnableConfig):
    """Loads relevant memories based on the current conversation history."""
    # Simple approach: use the last human message as the query
    last_human_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_message = msg.content
            break
    
    if last_human_message:
        relevant_memories = search_story_preferences.invoke(
            {"query": last_human_message}, config=config
        )
    else:
        relevant_memories = ["No conversation history yet."]
        
    # Ensure relevant_memories is always a list of strings
    if isinstance(relevant_memories, str):
        relevant_memories = [relevant_memories]
    elif not isinstance(relevant_memories, list):
        relevant_memories = ["Error retrieving memories."]
        
    return {"recall_memories": relevant_memories}

# Tool routing function
def should_continue_node(state: AgentState) -> Literal["tools", "agent"]:
    """Determines whether to continue with tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM made a tool call, route to the tool node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, continue to the agent node (or end if appropriate)
    return "agent" # In this simple loop, we always go back to the agent after loading memories

# --- Build the Graph --- 

builder = StateGraph(AgentState)

# Add nodes
builder.add_node("load_memories", load_memories_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

# Add edges
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")

# Conditional edge from agent to tools or back to agent (or END)
builder.add_conditional_edges(
    "agent",
    # Function to decide which path to take
    lambda state: "tools" if state["messages"][-1].tool_calls else "__end__",
    {
        "tools": "tools",
        "__end__": END
    }
)

# Edge from tools back to agent
builder.add_edge("tools", "agent")

# Set up memory
# The MemorySaver is responsible for persisting the state
# For in-memory use, this doesn't save across script runs unless configured with a persistent backend.
memory = MemorySaver()

# Compile the graph
story_agent_graph = builder.compile(checkpointer=memory)

# --- Example Usage (for testing) --- 

if __name__ == "__main__":
    print("Testing Storyteller Agent...")
    
    # Example conversation flow
    config_user_1 = {"configurable": {"user_id": "user_1", "thread_id": "thread_1"}}
    
    # Initial interaction
    print("\n--- Turn 1 --- ")
    inputs = {"messages": [HumanMessage(content="Hi! I love fantasy stories with dragons.")]}
    for event in story_agent_graph.stream(inputs, config=config_user_1):
        for key, value in event.items():
            print(f"Output from node 	'{key}':")
            print("---")
            print(value)
        print("\n")
        
    # Follow-up interaction
    print("\n--- Turn 2 --- ")
    inputs = {"messages": [HumanMessage(content="Can you tell me a short story?")]}
    for event in story_agent_graph.stream(inputs, config=config_user_1):
        for key, value in event.items():
            print(f"Output from node 	'{key}':")
            print("---")
            print(value)
        print("\n")

    # Check state
    # current_state = story_agent_graph.get_state(config_user_1)
    # print("\nCurrent State:", current_state)

