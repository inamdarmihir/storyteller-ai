# Langgraph Personalized Storyteller Agent

This project demonstrates a conversational agent with long-term memory capabilities built using Langgraph. The agent acts as a personalized storyteller, remembering user preferences (genres, characters, settings) across interactions to generate unique narratives.

## Features

- **Long-Term Memory:** Utilizes Langgraph and a vector store (e.g., ChromaDB or FAISS) to save and retrieve user preferences and past interactions.
- **Personalized Storytelling:** Generates stories tailored to the user based on remembered preferences.
- **Open-Source LLM:** Designed to work with open-source Large Language Models accessible via Ollama (e.g., Llama 3, Mistral).
- **Streamlit UI:** Provides a simple chat interface for interacting with the agent.

## Setup

1.  **Clone the repository (or extract the zip file).**
2.  **Install Ollama:** If you haven't already, install Ollama and pull a desired open-source model:
    ```bash
    # Example: Pull Llama 3
    ollama pull llama3
    ```
    Ensure the Ollama service is running.
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment (Optional):** If using specific API keys or configurations (though this example focuses on local Ollama), set them as environment variables.

## Running the Application

1.  **Start the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
2.  Open your browser to the provided local URL (usually `http://localhost:8501`).
3.  Interact with the storyteller agent! Tell it your preferences and ask for stories.

## Project Structure

- `agent.py`: Contains the core Langgraph agent logic, including memory management, LLM integration, and graph definition.
- `app.py`: Implements the Streamlit chat interface.
- `requirements.txt`: Lists necessary Python packages.
- `README.md`: This file.
- `todo.md`: Development checklist (can be removed).

