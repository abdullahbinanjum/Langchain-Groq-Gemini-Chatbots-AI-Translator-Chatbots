import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime # Needed for dynamic date in system prompt

# --- Configuration & Setup ---

# Load environment variables from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set Streamlit page configuration
st.set_page_config(page_title="🌌 Gemini Chatbot", page_icon="✨", layout="centered")

# Initialize chat history and temperature in session state
# st.session_state is crucial for maintaining state across reruns
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system_welcome", "content": "Hello! How can I assist you today?"}] # Initial welcome message
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7 # Default creativity setting

# --- Streamlit UI Components ---

st.title('🌌 Langchain AI Chatbot with Gemini Flash')

# Sidebar for controls and info
with st.sidebar:
    st.header("Settings & Info")
    st.markdown("---")
    st.write("This is an AI Chatbot powered by Google Gemini (Flash model) and Langchain.")

    # Temperature slider for creativity control
    st.subheader("AI Creativity")
    new_temperature = st.slider(
        "Adjust response creativity (Temperature)",
        min_value=0.0, max_value=1.0, value=st.session_state.temperature, step=0.05,
        help="Lower values mean more focused and deterministic responses. Higher values mean more creative and diverse responses."
    )
    # Update session state if slider value changes and trigger rerun
    if new_temperature != st.session_state.temperature:
        st.session_state.temperature = new_temperature
        # st.experimental_rerun() is used to immediately apply settings changes
        # It forces the script to rerun from top, picking up the new temperature
        st.experimental_rerun()

    st.markdown("---")
    st.info("💡 Tip: Ask about anything! E.g., 'Explain quantum physics in simple terms'.")

    # Clear chat history button
    if st.button("🚀 Clear Chat History"):
        st.session_state.messages = [{"role": "system_welcome", "content": "Hello! How can I assist you today?"}] # Reset with welcome
        st.success("Chat history cleared!")
        st.rerun() # Rerun to update the displayed messages

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "system_welcome":
        # Display initial welcome message outside of chat bubble
        st.markdown(f"**🤖 {message['content']}**")
    else:
        # Display user and AI messages in chat bubbles
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Main Chat Input ---
# Use st.chat_input for a modern chat-like input box
prompt_input = st.chat_input("Ask me anything...")

# --- Process User Input ---
if prompt_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Initialize LLM with current temperature from session state
    # This ensures the LLM's creativity setting is always up-to-date
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=st.session_state.temperature # Use the current temperature
    )

    # Define the prompt for the chatbot, now capable of handling conversation history
    # MessagesPlaceholder allows passing a list of messages directly
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"You are a friendly and helpful chatbot. Your responses should be concise and direct. Current date is {datetime.now().strftime('%Y-%m-%d')}."),
            MessagesPlaceholder(variable_name="chat_history"), # This placeholder will receive the past messages
            ("human", "{question}") # The current question from the user
        ]
    )

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    with st.chat_message("ai"):
        with st.spinner("AI is thinking... 🤔"):
            # Prepare chat history for the prompt
            # We convert session_state messages into the format expected by LangChain's MessagesPlaceholder
            # We skip the "system_welcome" message as it's not part of the actual conversation history sent to the LLM.
            # We also exclude the *very last* user message, as it's passed separately as "question".
            formatted_chat_history = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    formatted_chat_history.append(("human", msg["content"]))
                elif msg["role"] == "ai":
                    formatted_chat_history.append(("ai", msg["content"]))

            # The current prompt_input is already the last item added to st.session_state.messages as "user" role.
            # When using MessagesPlaceholder, we typically pass the *entire* past conversation (excluding the very last user turn which is `question`).
            # So, `chat_history_for_llm` should contain all messages *before* the current `prompt_input`.

            # Refined chat history preparation:
            # The prompt is constructed to take `chat_history` (all previous turns) and `question` (the current turn).
            # So, `st.session_state.messages` contains all previous turns + current turn.
            # We take all messages *except the very last one* for `chat_history`.
            # The last one is the `prompt_input` that will be mapped to `question`.

            # Filter out the initial 'system_welcome' and prepare the history for the LLM
            llm_chat_history = []
            for msg in st.session_state.messages[:-1]: # All messages EXCEPT the very last user input
                if msg["role"] == "user":
                    llm_chat_history.append(("human", msg["content"]))
                elif msg["role"] == "ai":
                    llm_chat_history.append(("ai", msg["content"]))


            response_stream = chain.stream({
                'chat_history': llm_chat_history,
                'question': prompt_input # The current question from the st.chat_input
            })
            full_response = st.write_stream(response_stream) # Stream output for better UX

    # Add AI response to chat history
    st.session_state.messages.append({"role": "ai", "content": full_response})