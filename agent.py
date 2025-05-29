"""
Core Langgraph agent logic for the Personalized Storyteller.

This module defines the agent's state, tools, nodes, edges, and graph
for managing long-term memory and generating personalized stories using
an open-source LLM via the Hugging Face Inference API.
"""

import os
import uuid
import faiss
from typing import List, TypedDict, Annotated, Sequence, Literal
from dotenv import load_dotenv
from requests.exceptions import JSONDecodeError # Import the specific error

# Load environment variables from .env file (optional, mainly for local testing)
load_dotenv()

from langchain_community.vectorstores import FAISS
# Use Hugging Face embeddings suitable for Inference API
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# Use Hugging Face Hub for the LLM
from langchain_community.llms import HuggingFaceHub

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# --- Configuration --- 
# Ensure HUGGINGFACEHUB_API_TOKEN is set in your environment or Streamlit secrets.
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# --- Critical Check for API Token --- 
if not HUGGINGFACEHUB_API_TOKEN:
    # In a deployed app, raising an error is often better than just warning.
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN environment variable not found. "
        "This is required to use the Hugging Face Inference API for embeddings and the LLM. "
        "Please ensure it is set in your environment (e.g., Streamlit secrets)."
    )

# --- Memory Setup --- 

# Using Hugging Face Inference API Embeddings
# Requires a HUGGINGFACEHUB_API_TOKEN
# Using a common sentence-transformer model available on HF Inference API
try:
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HUGGINGFACEHUB_API_TOKEN,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
except Exception as e:
    # Catch potential errors during embedding model initialization itself
    raise RuntimeError(f"Failed to initialize HuggingFaceInferenceAPIEmbeddings: {e}") from e

# FAISS vector store (in-memory)
# Note: For persistence across runs, you'd save/load the index.
# Using default FAISS setup for simplicity
# Check if the vector store file exists, load it, otherwise create a new one
FAISS_INDEX_PATH = "faiss_story_index"
vector_store = None # Initialize vector_store to None

try:
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index.")
        # Initialize with a dummy entry if creating new
        # This is where the JSONDecodeError was happening
        vector_store = FAISS.from_texts(["Initial dummy entry"], embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)

except JSONDecodeError as e:
    # Specific handling for JSONDecodeError during FAISS initialization
    error_message = (
        "Failed to create or load FAISS index due to a JSONDecodeError. "
        "This often means the Hugging Face Inference API did not return a valid JSON response, possibly due to:\n"
        "1. Invalid or missing HUGGINGFACEHUB_API_TOKEN.\n"
        "2. Network issues connecting to the Hugging Face API.\n"
        "3. Temporary problems with the Hugging Face Inference API or the specific model.\n"
        "4. Rate limits being exceeded.\n\n"
        "Please check your API token, network connection, and the Hugging Face status page. "
        f"Original error: {e}"
    )
    print(f"ERROR: {error_message}") # Log the detailed error
    raise RuntimeError(error_message) from e

except Exception as e:
    # Catch any other potential errors during FAISS setup
    raise RuntimeError(f"An unexpected error occurred during FAISS index setup: {e}") from e

# Ensure vector_store was successfully initialized
if vector_store is None:
    raise RuntimeError("FAISS vector store could not be initialized. Check previous errors.")

# --- Tool Definitions --- 

def get_user_id(config: RunnableConfig) -> str:
    """Retrieves the user ID from the runnable config."""
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id")
    if user_id is None:
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
    try:
        vector_store.add_documents([document])
        # Persist the updated index
        vector_store.save_local(FAISS_INDEX_PATH)
        return f"Preference saved: {preference}"
    except Exception as e:
        print(f"Error saving preference to vector store: {e}")
        return f"Error saving preference: {e}"

@tool
def search_story_preferences(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant user preferences based on the current conversation or query."""
    user_id = get_user_id(config)
    try:
        results = vector_store.similarity_search(
            query, k=3, filter={"user_id": user_id}
        )
        if not results:
            return ["No specific preferences found for this user."]
        return [doc.page_content for doc in results]
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return ["Error searching preferences."]

tools = [save_story_preference, search_story_preferences]
tool_node = ToolNode(tools)

# --- Agent State --- 

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    recall_memories: List[str]

# --- Nodes and Edges --- 

# Define the LLM using HuggingFaceHub
# Requires HUGGINGFACEHUB_API_TOKEN
# Using Meta's Llama 3 8B Instruct model via Inference API
try:
    llm = HuggingFaceHub(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512} # Adjust max_new_tokens as needed
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize HuggingFaceHub LLM: {e}. Check API token and model repo_id.") from e

# Define the prompt (remains largely the same, but ensure LLM understands tool use instructions)
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
            "Respond directly to the user. Do not simulate multi-turn conversations within a single response. "
            # Tool usage instructions might need refinement depending on the base LLM's capabilities
            "If you need to use a tool, format your request like this: \n"
            "Action: tool_name\nAction Input: query\nObservation: [Wait for tool result]\nFinal Answer: [Your final response after getting observation]"
            # Note: Direct tool binding with .bind_tools() might be less reliable with basic HuggingFaceHub LLM class.
            # We might need a more explicit tool handling mechanism or use a ChatModel wrapper if available.
            # For simplicity, we'll rely on the prompt instructions for now.
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Combine prompt and LLM
# Note: Direct tool binding with .bind_tools() might not work reliably with HuggingFaceHub.
# We will handle tool calls based on the LLM's text output if needed.
agent_runnable = prompt | llm

# Agent node function (modified for potentially manual tool handling)
def agent_node(state: AgentState, config: RunnableConfig):
    """Invokes the LLM to generate a response or decide on tool use."""
    recall_str = "\n".join(state["recall_memories"])
    try:
        response_text = agent_runnable.invoke(
            {"messages": state["messages"], "recall_memories": recall_str},
            config=config
        )
    except Exception as e:
        # Handle potential errors during LLM invocation
        print(f"Error invoking LLM: {e}")
        response_text = f"Sorry, I encountered an error trying to generate a response: {e}"
    
    # Basic check if the response indicates a tool call (based on prompt instructions)
    # This is a simplified approach. A more robust method would parse structured output or use ReAct logic.
    ai_message = AIMessage(content=response_text)
    if "Action: save_story_preference" in response_text or "Action: search_story_preferences" in response_text:
        # Placeholder for potential future structured tool call parsing
        # For now, we assume the LLM might try to call tools via text, 
        # but the graph structure handles explicit tool calls better if the LLM supports it.
        # If using bind_tools becomes feasible, this logic changes.
        print("LLM response suggests tool use (text-based). Relying on graph structure for actual calls if LLM supports bind_tools.")
        # Pass the raw response for now. If bind_tools works, it will populate tool_calls.
        # If bind_tools doesn't work with HuggingFaceHub, this node needs more logic to parse and create ToolMessage.
        pass 
        
    return {"messages": [ai_message]}

# Memory loading node function (remains the same)
def load_memories_node(state: AgentState, config: RunnableConfig):
    """Loads relevant memories based on the current conversation history."""
    last_human_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_message = msg.content
            break
    
    if last_human_message:
        # Invoke the tool directly, handling potential list/string return types
        try:
            relevant_memories_result = search_story_preferences.invoke(
                {"query": last_human_message}, config=config
            )
            if isinstance(relevant_memories_result, str):
                 relevant_memories = [relevant_memories_result]
            elif isinstance(relevant_memories_result, list):
                 relevant_memories = relevant_memories_result
            else:
                 print(f"Warning: Unexpected type from search_story_preferences: {type(relevant_memories_result)}")
                 relevant_memories = ["Error: Unexpected memory format."]
        except Exception as e:
            print(f"Error invoking search_story_preferences tool: {e}")
            relevant_memories = [f"Error retrieving memories: {e}"]
    else:
        relevant_memories = ["No conversation history yet."]
        
    return {"recall_memories": relevant_memories}

# Tool routing function (remains the same, relies on .tool_calls attribute)
def should_continue_node(state: AgentState):
    """Determines whether to route to tools or end the interaction."""
    messages = state["messages"]
    if not messages:
        return END # Should not happen with START node, but safety check
    last_message = messages[-1]
    # If the LLM attached tool calls, route to tools
    # Check for the attribute *and* if it's non-empty
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Otherwise, end the conversation turn
    return END

# --- Build the Graph --- 

builder = StateGraph(AgentState)

# Add nodes
builder.add_node("load_memories", load_memories_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

# Add edges
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")

# Conditional edge from agent
builder.add_conditional_edges(
    "agent",
    should_continue_node, # Use the function to decide the next step
    {
        "tools": "tools",
        END: END
    }
)

# Edge from tools back to agent
builder.add_edge("tools", "agent")

# Set up memory (remains the same)
memory = MemorySaver()

# Compile the graph
story_agent_graph = builder.compile(checkpointer=memory)

# --- Example Usage (for local testing) --- 

if __name__ == "__main__":
    print("Testing Storyteller Agent (Hugging Face API)...")
    
    # The check for the token is now done earlier, 
    # so this block will only run if the token exists and FAISS/LLM initialized.
    try:
        # Example conversation flow
        config_user_1 = {"configurable": {"user_id": "user_1_hf_test", "thread_id": "thread_hf_test_1"}}
        
        # Initial interaction
        print("\n--- Turn 1 --- ")
        inputs = {"messages": [HumanMessage(content="Hi! I love sci-fi stories set in space.")]}
        for event in story_agent_graph.stream(inputs, config=config_user_1):
            for key, value in event.items():
                print(f"Output from node '{key}':")
                print("---")
                print(value)
            print("\n")
            
        # Follow-up interaction
        print("\n--- Turn 2 --- ")
        inputs = {"messages": [HumanMessage(content="Tell me a short story about exploring a new planet.")]}
        for event in story_agent_graph.stream(inputs, config=config_user_1):
            for key, value in event.items():
                print(f"Output from node '{key}':")
                print("---")
                print(value)
            print("\n")

        # Check state (optional)
        # try:
        #     current_state = story_agent_graph.get_state(config_user_1)
        #     print("\nCurrent State:", current_state)
        # except Exception as e:
        #     print(f"Error getting state: {e}")
            
    except Exception as e:
        print(f"\n--- Test Run Failed --- ")
        print(f"An error occurred during the test run: {e}")
        print("This might be due to configuration issues (like API token) or runtime errors.")


