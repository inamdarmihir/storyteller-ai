"""
Streamlit application for the Personalized Storyteller Agent.

This script creates a chat interface allowing users to interact with the
Langgraph agent defined in agent.py.
"""

import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage

# Assuming agent.py is in the same directory
from agent import story_agent_graph

st# --- Page Configuration ---
st.set_page_config(page_title="Personalized Storyteller", page_icon="📖")
st.title("📖 Personalized Storyteller Agent")
st.caption("An agent that remembers your preferences to tell unique stories (using Langgraph & Ollama)")

# --- Session State Initialization ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hi! I am your personalized storyteller. Tell me what kind of stories you like (e.g., genres, characters, settings) or ask me to tell you a story!")]

# Initialize user ID
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# Initialize thread ID (a unique ID for this specific chat session)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- Helper Functions ---

def get_agent_response(user_input: str):
    """Invokes the agent graph and streams the response."""
    config = {
        "configurable": {
            "user_id": st.session_state.user_id,
            "thread_id": st.session_state.thread_id, # Use session-specific thread ID
        }
    }
    
    # Prepare input for the graph
    inputs = {"messages": [HumanMessage(content=user_input)]}
    
    response_content = ""
    final_ai_message = None

    st.write(f"_Thinking... (User ID: {st.session_state.user_id}, Thread ID: {st.session_state.thread_id})_ ")
    # Stream events from the graph
    try:
        # Use stream to get intermediate steps if needed, or invoke for final result
        # For a chat app, getting the final AI message is often sufficient.
        # Using stream here to potentially show tool calls in the future, but extracting final message.
        for event in story_agent_graph.stream(inputs, config=config, stream_mode="values"):
            # The final response is typically in the 'agent' key's output messages
            if "messages" in event:
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage):
                    final_ai_message = last_message

        if final_ai_message:
            response_content = final_ai_message.content
        else:
            response_content = "Sorry, I encountered an issue generating a response." 
            # You could add more detailed error logging here based on stream events

    except Exception as e:
        st.error(f"An error occurred: {e}")
        response_content = "An error occurred while processing your request."

    return response_content

# --- Chat Interface --- 

# Display chat messages from history
for message in st.session_state.messages:
    avatar = "👤" if isinstance(message, HumanMessage) else "📖"
    with st.chat_message(message.type):
        st.markdown(message.content)

# React to user input
if prompt := st.chat_input("Tell me your preferences or ask for a story..."):
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    # Display user message in chat message container
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant", avatar="📖"):
        with st.spinner("Generating story..."):
            response = get_agent_response(prompt)
            st.markdown(response)
    
    # Add agent response to chat history
    st.session_state.messages.append(AIMessage(content=response))

